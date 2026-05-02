from openai import OpenAI
import streamlit as st
import torch
import numpy as np
from PIL import Image

from build_model import build_model
 
@st.cache_resource
def load_model(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt   = torch.load(ckpt_path, map_location=device)
    
    model  = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device

@st.cache_resource
def get_llm_client():
  client = OpenAI(base_url="https://api.studio.nebius.ai/v1/",
                  api_key=st.secrets["NEBIUS_PLATE_API_KEY"])
  return client

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr   = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor

def parse_plated(user_message: str,
                 client) -> list[int]:
  response = client.chat.completions.create(
      model="meta-llama/Meta-Llama-3.1-8B-Instruct",
      messages=[
          {
              "role": "system",
              "content": (
                  "Extract the total plated calories (kcal) and total plated protein (grams) from the user message. "
                  "ALWAYS return in this exact format regardless of the order they appear in the message: "
                  "KCAL,PROTEIN "
                  "Examples: "
                  "'Patient had 876 kcal and 87g protein' → '876,87' "
                  "'Patient had 34g protein and 540 kcal' → '540,34' "
                  "If either value is missing return 0: '540,0' or '0,34' "
                  "Return ONLY the two numbers separated by a comma. No other text."
              )
          },
          {"role": "user", "content": user_message}
      ],
      max_tokens=100,
  )

  raw = response.choices[0].message.content.strip()

  content_list = []
  for x in raw.split(','):
    x = x.strip()
    if x == "null" or x =="" or int(float(x)) < 0:
       content_list.append(0)
    else:
       content_list.append(int(float(x)))

  return content_list

def generate_response(plated_amounts: list[int],
                      predicted_waste_kcal: int,
                      predicted_waste_protein: int,
                      client) -> str:
  consumed_kcal = plated_amounts[0] - predicted_waste_kcal
  consumed_pro = plated_amounts[1] - predicted_waste_protein

  if plated_amounts[0] == 0:
    percent_kcal = None
  else:
    percent_kcal = (consumed_kcal / plated_amounts[0]) * 100

  if plated_amounts[1] == 0:
    percent_pro = None
  else:
    percent_pro = (consumed_pro / plated_amounts[1]) * 100

  prompt = f"""
  A patient's plate intake has been estimated by a computer vision model.
  Use ONLY the numbers below. Do NOT recalculate or derive any other numbers.

  FACTS (do not modify these):
  - Patient was Plated: {plated_amounts[0]:.0f} kcal and {plated_amounts[1]:.0f}g protein
  - Remaining on plate: {predicted_waste_kcal:.0f} kcal and {predicted_waste_protein:.1f}g protein
  - Consumed: {consumed_kcal:.0f} kcal ({percent_kcal}% of plated) and {consumed_pro:.1f}g protein ({percent_pro:.1f}% of plated)

  Write a 3-4 sentence clinical intake note using ONLY the FACTS above.
  Do NOT perform any calculations.
  Do NOT add any numbers that are not in the FACTS section.
  Do NOT add preamble like "Here is a note".
  Output the clinical note directly.
  """
  response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": (
    "You are a clinical nutrition assistant writing intake notes. "
    "You must use ONLY the numbers explicitly provided to you. "
    "Never calculate, derive, or invent numbers. "
    "Output clinical notes directly with zero preamble.")},
        {"role": "user",   "content": prompt}
    ],
    max_tokens=200,)

  return response.choices[0].message.content.strip()

# --------------------------------------------------------
#  streamlit UI
# --------------------------------------------------------

def main():
    st.title("Plate Intake Estimator")
    st.caption("Clinical tool for estimating patient calorie intake from plate waste photos")

    model, device = load_model("models/efficientnet_b3_UF6_best.pt")
    client        = get_llm_client()

    # user inputs
    user_message = st.text_area(
        "Clinical context",
        placeholder="e.g. My patient had 420 kcal and 35 g protein plated for lunch."
    )
    image_file = st.file_uploader(
        "Photo of remaining plate",
        type=["jpg", "jpeg", "png"]
    )

    if st.button("Estimate Intake"):

        # edge case — missing inputs
        if not user_message:
            st.warning("Please describe the meal content (plated kcal and protein).")
            st.stop()
        if not image_file:
            st.warning("Please upload a photo of the remaining plate.")
            st.stop()

        # parse plated kcal from message
        plated_amounts = parse_plated(user_message, client)

        if plated_amounts[0] is None:
            st.warning(
                "Could not find a plated calorie amount in your message. "
                "Please include it, e.g. '420 kcal was plated'."
            )
            st.stop()

        # run model
        with st.spinner("Analysing plate..."):
            image  = Image.open(image_file)
            tensor = preprocess_image(image).to(device)

            with torch.no_grad():
                preds = model(tensor).squeeze(0).cpu().numpy()  # (2,)

            remaining_kcal    = float(preds[0])
            remaining_protein = float(preds[1])

        # edge case — model predicts more than plated
        if remaining_kcal > plated_amounts[0]:
            st.warning(
                f"Model estimated {remaining_kcal:.0f} kcal remaining which exceeds "
                f"the plated amount of {plated_amounts[0]:.0f} kcal. "
                "Image may be ambiguous — please verify manually."
            )

        # generate clinical response
        with st.spinner("Generating clinical summary..."):
            response = generate_response(
               plated_amounts = plated_amounts,
               predicted_waste_kcal = remaining_kcal,
               predicted_waste_protein = remaining_protein,
               client = client)

        # display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded plate", use_container_width=True)
        with col2:
            consumed_kcal = max(0, plated_amounts[0] - remaining_kcal)
            consumed_pro = max(0, plated_amounts[1] - remaining_protein)

            st.metric("Plated Kcal",    f"{plated_amounts[0]:.0f} kcal")
            st.metric("Plated Protein",    f"{plated_amounts[1]:.0f} g")

            st.metric("Est. Remaining Kcal", f"{remaining_kcal:.0f} kcal")
            st.metric("Est. Remaining Protein", f"{remaining_protein:.0f} g")

            st.metric("Consumed Kcal",  f"{consumed_kcal:.0f} kcal")
            st.metric("Consumed Protein",  f"{consumed_pro:.0f} g")           

        st.divider()
        st.subheader("Clinical Summary")
        st.write(response)

        # disclaimer
        st.caption(
            "⚠️ This is a research tool. Model estimates carry uncertainty "
            "and should not replace clinical judgment."
        )

if __name__ == "__main__":
    main()


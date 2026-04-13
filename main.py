import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import qrcode
import re  # Added for input validation
from io import BytesIO
from datetime import datetime
from fpdf import FPDF

# 1. Page Configuration
st.set_page_config(
    page_title="PrecisionDerm AI | Medical Portal",
    layout="wide",
    page_icon="🏥"
)

# 2. MAIN TITLE
st.markdown("""
<h1 style='text-align: center;'>🏥 PrecisionDerm AI</h1>
<p style='text-align:center; color:#6d28d9; font-size:16px;'>
Advanced AI-Powered Skin Analysis for Early and Accurate Detection
</p>
""", unsafe_allow_html=True)

# 3. CSS 
st.markdown("""
<style>
/*Global Reset*/
header { visibility: hidden; }
.stDeployButton { display: none; }
footer { visibility: hidden; }
    
/* Main Dashboard Background*/
.stApp { 
    background: radial-gradient(circle at top right, #f5f3ff, #ffffff 70%) !important;
}

/* Sidebar Styling */
[data-testid="stSidebar"] { 
    background: linear-gradient(180deg, #4c1d95 0%, #1e1b4b 100%) !important;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown { 
    color: #f5f3ff !important; 
}

/* Sidebar textboxes: Patient Name & Age */
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {
    background-color: #ffffff !important;  
    border: 1px solid #ddd6fe !important;
    border-radius: 10px !important;
    color: #4c1d95 !important;  
    font-weight: 600 !important;
    padding: 10px !important;
}
[data-testid="stSidebar"] input::placeholder, [data-testid="stSidebar"] textarea::placeholder {
    color: #a78bfa !important;
    opacity: 0.7;
}

/* File Uploader Gradient */
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, #4c1d95 0%, #1e1b4b 100%) !important;
    border: 2px dashed #a78bfa !important;
    color: white !important;
}
[data-testid="stFileUploaderDropzone"] div div span {
    color: white !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    fill: white !important;
}

/* Large Result Card */
.result-card { 
    padding: 40px !important; 
    border-radius: 25px !important; 
    background-color: #ffffff !important; 
    border-left: 10px solid #7c3aed !important; 
    box-shadow: 0 15px 35px rgba(76, 29, 149, 0.1) !important;
    min-height: 220px !important; 
    display: flex;
    flex-direction: column;
    justify-content: center;
}

/* Top 3 Results*/
.top-result-box {
    border: 2px solid #a78bfa !important; 
    padding: 15px !important; 
    border-radius: 12px !important; 
    margin-bottom: 12px !important;
    background: white !important;
    height: 100px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.top-result-name { color: #6d28d9 !important; font-weight: 800; font-size: 14px; }
.top-result-conf { color: #7c3aed !important; font-weight: 600; font-size: 13px; }

/*PDF Button*/
div.stDownloadButton > button { 
    width: 100% !important; 
    background: linear-gradient(90deg, #7c3aed 0%, #4c1d95 100%) !important; 
    color: white !important;
    padding: 18px !important;
    border-radius: 15px !important;
    font-weight: 700 !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
}

/* Gradient Headers */
h1, h2, h3, .gradient-header { 
    background: linear-gradient(90deg, #6d28d9, #a78bfa);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
}

/* FileUploader / Image Caption Gradient */
[data-testid="stFileUploader"] label, [data-testid="stImage"] p {
    background: linear-gradient(90deg, #6d28d9, #a78bfa);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
}

/* Checkbox Gradient */
.stCheckbox label p {
    background: linear-gradient(90deg, #4c1d95, #7c3aed) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
}

/* Clinical Analysis Gradient */
.gradient-label {
    background: linear-gradient(90deg, #4c1d95, #7c3aed) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    display: block !important;
}
.gradient-text-body {
    background: linear-gradient(90deg, #6d28d9, #8b5cf6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 600 !important;
    line-height: 1.6 !important;
    display: block !important;
    margin-bottom: 15px !important;
}

/* Scanner Container */
.waiting-container { 
    height: 180px !important;   
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: rgba(255, 255, 255, 0.6) !important;
    border: 2px dashed #8b5cf6 !important;
    border-radius: 25px;
}
.waiting-text { 
    background: linear-gradient(90deg, #5b21b6, #8b5cf6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 16px !important;   
    font-weight: 900; 
    letter-spacing: 1px;
    text-align: center;
}

/* Medical Disclaimer */
.medical-disclaimer {
    background: #1e1b4b !important;
    border-left: 10px solid #8b5cf6 !important;
    padding: 30px;
    border-radius: 20px;
    margin-top: 40px;
}
.medical-disclaimer b { color: #a78bfa !important; }
.medical-disclaimer p { color: #cbd5e1 !important; }

</style>
""", unsafe_allow_html=True)


# PDF Generator
def generate_pdf_report(result, conf, details, p_name, p_age, p_address, observations):
    pdf = FPDF()
    pdf.add_page()
    def clean_text(text):
        return str(text).encode('cp1252', 'ignore').decode('cp1252') if text else ""
    
    pdf.set_font("Arial", 'B', 20); pdf.set_text_color(109, 40, 217) 
    pdf.cell(200, 15, txt=clean_text("PrecisionDerm AI - Diagnostic Report"), ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_fill_color(245, 243, 255); pdf.set_font("Arial", 'B', 12); pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, txt=" Patient Information", ln=True, fill=True)
    pdf.set_font("Arial", size=11)
    
    # Format: Name, Age, Address on separate lines
    pdf.cell(0, 8, txt=clean_text(f"Name: {p_name if p_name else 'Not Provided'}"), ln=True) 
    pdf.cell(0, 8, txt=f"Age: {p_age if p_age else 'N/A'}", ln=True)
    pdf.multi_cell(0, 8, txt=clean_text(f"Address: {p_address if p_address else 'Not Provided'}"))
    pdf.ln(5)
    
    active_obs = [k for k, v in observations.items() if v]
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, txt="Clinical Observations (Self-Reported):", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 7, txt=clean_text(", ".join(active_obs) if active_obs else "No specific symptoms reported."))
    pdf.ln(5)
    
    pdf.set_fill_color(245, 243, 255); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt=" AI Diagnostic Results", ln=True, fill=True)
    pdf.set_font("Arial", 'B', 14); pdf.set_text_color(124, 58, 237)
    pdf.cell(200, 10, txt=clean_text(f"Condition: {result}"), ln=True)
    pdf.set_font("Arial", size=11); pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 8, txt=f"Confidence Score: {conf:.2f}%", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 10); pdf.cell(0, 8, txt="Detailed Analysis:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt=clean_text(f"Description: {details['Description']}"))
    pdf.ln(2)
    pdf.multi_cell(0, 6, txt=clean_text(f"Clinical Note: {details['Clinical Note']}"))
    pdf.ln(2)
    pdf.multi_cell(0, 6, txt=clean_text(f"Recommended Action: {details['Action']}"))
    
    pdf.ln(15)
    pdf.set_draw_color(124, 58, 237)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10); pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 8, txt="MEDICAL DISCLAIMER:", ln=True)
    pdf.set_font("Arial", 'I', 9); pdf.set_text_color(80, 80, 80)
    disclaimer_text = ("This report is generated by an AI screening tool and is intended for informational purposes only. It is NOT a clinical diagnosis. Always seek the advice of a qualified dermatologist for final clinical assessment.")
    pdf.multi_cell(0, 5, txt=clean_text(disclaimer_text))
    
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# Disease Data
disease_info = {
    'Actinic keratoses': {
        'Description': "These are rough, sandpaper-like scaly patches caused by cumulative ultraviolet (UV) radiation damage. They typically manifest on sun-exposed regions such as the scalp, face, and ears. The texture is often more noticeable than the visual color.",
        'Clinical Note': "Medically classified as an intraepidermal neoplasm (precancerous). If left untreated, approximately 10% progress into invasive Squamous Cell Carcinoma (SCC). They serve as a primary indicator of field cancerization.",
        'Action': "Clinical evaluation is mandatory. Management options include localized Cryotherapy (liquid nitrogen), field therapy with topical 5-Fluorouracil, or Photodynamic Therapy (PDT). Strict adherence to broad-spectrum SPF 50+ is required."
    },
    'Basal cell carcinoma': {
        'Description': "Basal Cell Carcinoma (BCC) is the most prevalent malignancy in humans. It frequently presents as a slow-growing, pearly or waxy nodule with prominent telangiectasia (visible tiny blood vessels) and may occasionally ulcerate or bleed.",
        'Clinical Note': "Originating from the basal layer of the epidermis, BCC is locally destructive and can invade underlying cartilage or bone if neglected. While it rarely metastasizes to distant organs, it requires complete clearance to prevent recurrence.",
        'Action': "Surgical intervention is the standard of care. Mohs Micrographic Surgery is recommended for high-risk facial areas to ensure clear margins while preserving healthy tissue. Periodic post-treatment skin surveillance is essential."
    },
    'Benign keratosis': {
        'Description': "Technically known as Seborrheic Keratoses, these are non-cancerous epidermal growths. They often exhibit a 'stuck-on' appearance with a waxy, verrucous (wart-like) texture, ranging in color from light tan to deep black.",
        'Clinical Note': "These are benign proliferations of keratinocytes and carry zero malignant potential. However, they are frequently confused with Melanoma by patients due to their dark pigmentation and irregular texture.",
        'Action': "No clinical intervention is medically required. If the lesion becomes symptomatic (itchy, inflamed, or snagging on clothing), it can be removed via shave excision or curettage for symptomatic relief or cosmetic preference."
    },
    'Dermatofibroma': {
        'Description': "Firm, fibrohistiocytic nodules that typically occur on the extremities, particularly the lower legs. They are characterized by a 'button-like' feel and range from flesh-colored to hyperpigmented reddish-brown.",
        'Clinical Note': "Considered a benign reactive process often triggered by minor trauma like insect bites or ingrown hairs. A hallmark diagnostic indicator is the 'Dimple Sign'—the lesion retracts inward when pinched laterally.",
        'Action': "These are harmless and generally do not require removal. If the diagnosis is clinically uncertain or the lesion becomes painful, a punch biopsy may be performed. Surgical excision is elective but will result in a permanent scar."
    },
    'Melanoma': {
        'Description': "Melanoma is a highly aggressive malignancy arising from melanocytes. It is identified by the ABCDE criteria: Asymmetry, Border irregularity, Color variegation, Diameter >6mm, and Evolving nature. It can appear as a new spot or a change in an existing mole.",
        'Clinical Note': "Melanoma is the most lethal form of skin cancer due to its high potential for lymphatic and hematogenous metastasis. Survival rates are directly correlated with the 'Breslow Depth' (how deep the tumor has invaded).",
        'Action': "URGENT SURGICAL PRIORITY. Requires immediate wide-local excision and potentially a Sentinel Lymph Node Biopsy (SLNB) depending on depth. Specialist multidisciplinary care (Dermatology and Oncology) is critical."
    },
    'Melanocytic nevi': {
        'Description': "Commonly known as moles, these are benign organized clusters of melanocytes. They are typically uniform in color (tan to brown), possess regular, well-defined borders, and remain stable over long periods.",
        'Clinical Note': "While individual nevi are benign, a high total mole count (specifically >50) or the presence of 'Dysplastic' (atypical) nevi serves as a significant phenotypic marker for an increased risk of developing cutaneous melanoma.",
        'Action': "Routine monitoring via monthly self-skin examinations is advised. Any mole exhibiting the 'Ugly Duckling' sign (looking different from others) or changing in size, shape, or color should be evaluated via dermoscopy."
    },
    'Vascular lesions': {
        'Description': "A category of lesions arising from the cutaneous vasculature, including Cherry Angiomas, Hemangiomas, and Telangiectasias. They present as bright red, purple, or blue macules or papules.",
        'Clinical Note': "These represent benign proliferations of endothelial cells or structural abnormalities of blood vessels. They are non-malignant but can occasionally bleed if subjected to physical trauma.",
        'Action': "Treatment is entirely elective. For patients seeking cosmetic improvement, vascular-specific lasers (such as Pulsed Dye Laser or KTP) or electrodessication provide excellent results with minimal scarring."
    }
}


# Model Loading
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('best_skin_model.h5')

try:
    model = load_my_model()
except:
    model = None
class_names = list(disease_info.keys())

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2807/2807575.png", width=60)
    st.title("PrecisionDerm AI")
    st.divider()
    
    # Validation Logic
    p_name = st.text_input("Patient Name", placeholder="Enter Patient Name")
    if p_name and any(char.isdigit() for char in p_name):
        st.error("❌ Name should only contain letters!")
    
    p_age = st.text_input("Age", placeholder="Enter Patient Age")
    if p_age:
        if not p_age.isdigit():
            st.error("❌ Age should only contain numbers!")
        elif not (2 <= len(p_age) <= 3):
            st.error("❌ Age must be a 2 or 3 digit number!")

    # Address Box
    p_address = st.text_area("Patient Address", placeholder="Enter Street, City, State", height=100)
    
    st.divider()
    
    qr = qrcode.make("https://www.google.com/maps/search/dermatologist+near+me")
    buf = BytesIO(); qr.save(buf)
    st.image(buf, use_container_width=True)
    st.markdown("<p style='text-align:center; color:white;'>Scan to find a nearby dermatologist</p>", unsafe_allow_html=True)

# Main UI
st.divider()
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    uploaded_file = st.file_uploader("📁 Upload Skin Lesion Image", type=["jpg", "png", "jpeg"], label_visibility="visible")

    if not uploaded_file:
        st.markdown("""
        <div style='background: linear-gradient(180deg, #4c1d95 0%, #1e1b4b 100%);
                    color: white; padding:12px; border-radius:10px; margin-top:10px;
                    font-weight:500; font-size:14px;'>
            <b>📌 User Guidance:</b>
            <ul style='margin-top:5px;'>
                <li>Upload a clear, high-resolution image of the skin lesion.</li>
                <li>Ensure good lighting and avoid shadows or blurring.</li>
                <li>The lesion should be fully visible; avoid covering it with fingers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True, caption="Analyzed Specimen Image")

        obs_dict = {'Itching/Irritation': False, 'Bleeding/Crust': False, 
                    'Rapid Size Change': False, 'Color Irregularity': False}

        st.markdown(f"""
        <div style='background: linear-gradient(180deg, #4c1d95 0%, #1e1b4b 100%);
                    color:white; padding:10px; border-radius:10px; margin-top:10px; font-weight:600;' >
        <b>Image Info:</b> {uploaded_file.name}<br>
        <b>Uploaded:</b> {datetime.now().strftime('%d-%m-%Y %H:%M')}<br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:10px; font-weight:bold;'>📋 Clinical Observations:</div>", unsafe_allow_html=True)
        c1_obs, c2_obs = st.columns(2)
        with c1_obs: 
            obs_dict['Itching/Irritation'] = st.checkbox("Itching / Irritation")
            obs_dict['Bleeding/Crust'] = st.checkbox("Bleeding / Crust")
        with c2_obs: 
            obs_dict['Rapid Size Change'] = st.checkbox("Rapid Size Change")
            obs_dict['Color Irregularity'] = st.checkbox("Color Irregularity")
            
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0] * 100

        top_3_idx = np.argsort(preds)[-3:][::-1]

        st.markdown("<h3 class='gradient-header'> Top 3 Possible Conditions</h3>", unsafe_allow_html=True)

        t_cols = st.columns(3)

        for idx, i in enumerate(top_3_idx):
            with t_cols[idx]:
                disease_name = class_names[i]
                confidence = preds[i]

                st.markdown(f"""
                <div class="top-result-box">
                    <span class="top-result-name">{disease_name}</span>
                    <span class="top-result-conf">Conf: {confidence:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)

with col2:
    if not uploaded_file:
        st.markdown("""
        <div class="waiting-container">
            <p class="waiting-text">📡 SCANNER READY</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='background: linear-gradient(180deg, #4c1d95 0%, #1e1b4b 100%);
                    color: white; padding:12px; border-radius:10px; margin-top:10px;
                    font-weight:500; font-size:14px; line-height:1.5;' >
            <b>📌 Data Privacy Note:</b><br>
            1. All uploaded images and patient data are processed locally.<br>
            2. No images are stored permanently.<br>
            3. Data is deleted automatically after the session.
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        main_idx = top_3_idx[0]
        main_res = class_names[main_idx]
        main_conf = preds[main_idx]

        st.markdown(f"""
        <div class="result-card">
            <p style='color: #7c3aed; font-size: 13px; font-weight: bold; margin:0;'>AI DETECTION</p>
            <h2 style='margin:0;'>{main_res}</h2>
            <p style='color:#4c1d95; font-weight:bold;'>Confidence: {main_conf:.2f}%</p>
        </div>""", unsafe_allow_html=True)

        with st.expander("📄 Clinical Analysis", expanded=True):
            info = disease_info.get(main_res, {})
            st.markdown(f"""
            <p class="gradient-label">Description:</p>
            <p class="gradient-text-body">{info.get('Description', 'No description available.')}</p>
            <p class="gradient-label">Clinical Note:</p>
            <p class="gradient-text-body">{info.get('Clinical Note', 'No clinical notes available.')}</p>
            <p class="gradient-label">Action Required:</p>
            <p class="gradient-text-body">{info.get('Action', 'Consult a specialist for further advice.')}</p>
            """, unsafe_allow_html=True)

        st.divider()
        pdf_bytes = generate_pdf_report(main_res, main_conf, disease_info[main_res], p_name, p_age, p_address, obs_dict)
        st.download_button("📥 Download PDF Report", data=pdf_bytes, file_name=f"PrecisionDerm_{p_name}.pdf")

st.markdown("""
<div class="medical-disclaimer">
    <b>⚠️ MEDICAL DISCLAIMER</b><br>
    <p>This report is generated by an AI screening tool and is NOT a substitute for professional medical diagnosis. 
    Always seek the advice of a qualified dermatologist for final clinical assessment.</p>
</div>
""", unsafe_allow_html=True)
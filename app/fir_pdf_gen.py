from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from fpdf import FPDF
from pydantic import BaseModel
import os
import uuid
import httpx
import cloudinary
import cloudinary.uploader

cloudinary.config( 
    cloud_name = "dgdxa7qqg", 
    api_key = "376418913322648", 
    api_secret = "ut-74eisi_NAFxfrEUDhER2szgM",  # Your Cloudinary API secret
    secure=True
)
router = APIRouter()

class FIRPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "FIRST INFORMATION REPORT (FIR)", align="C", ln=True)
        self.ln(10)

def generate_fir_pdf(data: dict) -> str:
    pdf = FIRPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # Add content
    pdf.cell(0, 10, f"Book No.: {data['book_no']}", ln=True)
    pdf.cell(0, 10, f"Form No.: {data['form_no']}", ln=True)
    pdf.cell(0, 10, f"Police Station: {data['police_station']}", ln=True)
    pdf.cell(0, 10, f"District: {data['district']}", ln=True)
    pdf.cell(0, 10, f"Date and Hour of Occurrence: {data['date_hour_occurrence']}", ln=True)
    pdf.cell(0, 10, f"Date and Hour when Reported: {data['date_hour_reported']}", ln=True)
    pdf.cell(0, 10, f"Name and Residence of Informer/Complainant: {data['informer_name']}", ln=True)
    pdf.multi_cell(0, 10, f"Brief Description of Offense (with Section) and Property Carried Off (if any): {data['description_offense']}")
    pdf.cell(0, 10, f"Place of Occurrence and Distance/Direction from Police Station: {data['place_occurrence']}", ln=True)
    pdf.cell(0, 10, f"Name and Address of the Criminal: {data['criminal_name']}", ln=True)
    pdf.multi_cell(0, 10, f"Steps Taken Regarding Investigation/Explanation of Delay: {data['investigation_steps']}")
    pdf.cell(0, 10, f"Date and Time of Dispatch from Police Station: {data['dispatch_time']}", ln=True)
    pdf.cell(0, 10, f"Signature of Writer: ..............................", ln=True)
    
    temp_file = f"FIR_Report_{uuid.uuid4().hex}.pdf"
    pdf.output(temp_file)

    # Upload the PDF to Cloudinary
    try:
        print(f"Uploading file: {temp_file}")  # Debug print statement
        response = cloudinary.uploader.upload(
            temp_file,
            resource_type="raw",  # PDFs are treated as raw files in Cloudinary
            folder="fir_reports/"
        )
        os.remove(temp_file)  # Clean up the local file

        # Generate a downloadable and viewable URL
        view_url = response['secure_url']
        download_url = f"{view_url}?attachment=true"
        return {"view_url": view_url, "download_url": download_url}
    except Exception as e:
        os.remove(temp_file)  # Clean up in case of failure
        print(f"Error uploading to Cloudinary: {str(e)}")  # Debug print statement
        raise HTTPException(status_code=500, detail=f"Error uploading to Cloudinary: {str(e)}")
        
class FIRDetails(BaseModel):
    book_no: str
    form_no: str
    police_station: str
    district: str
    date_hour_occurrence: str
    date_hour_reported: str
    informer_name: str
    description_offense: str
    place_occurrence: str
    criminal_name: str
    investigation_steps: str
    dispatch_time: str

# Function to get LawGPT response
async def get_lawgpt_response(description_offense: str) -> str:
    """
    Sends the description_offense to an external service and retrieves the response.
    """
    url = "https://aniudupa-ani.hf.space/chat/"  # Replace with the actual URL
    try:
        # Construct the question to send to the LawGPT service
        question = f"Based on this incident: '{description_offense}', please provide a concise description of the offense, including the sections of the Indian Penal Code that apply."

        # Make the API request to LawGPT
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"question": question, "chat_history": "what"})
            response.raise_for_status()
            data = response.json()
            concise_description = data.get("answer", "").split("\n")[0]
            return concise_description

    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e.response.status_code}, {e.response.text}")  # Debug print statement
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        print(f"Failed to get response from LawGPT: {str(e)}")  # Debug print statement
        raise HTTPException(status_code=500, detail=f"Failed to get response from LawGPT: {str(e)}")

# Endpoint to generate FIR
@router.post("/")
async def generate_fir(details: FIRDetails):
    try:
        print(f"Received FIR details: {details}")  # Debug print statement
        detailed_offense = await get_lawgpt_response(details.description_offense)
        details.description_offense = detailed_offense
        urls = generate_fir_pdf(details.dict())
        return {
            "message": "FIR PDF generated successfully!",
            "view_url": urls["view_url"],  # Cloudinary view URL
            "download_url": urls["download_url"]  # Cloudinary download URL
        }
    except Exception as e:
        print(f"Error generating FIR: {str(e)}")  # Debug print statement
        raise HTTPException(status_code=500, detail=str(e))
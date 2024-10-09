import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from typing import List, Dict, Callable, Optional, Union
import re
from urllib.parse import urlparse
import pymupdf
import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import inspect
from openai import OpenAI
import uuid



# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn")


load_dotenv(find_dotenv(".env"))

# Log environment variables
logger.info("Environment variables:")
for key, value in os.environ.items():
    logger.info(f"{key}: {value}")

app = FastAPI()

# Add CORS middleware (for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://localhost:3000", "http://127.0.0.1:3000"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # You can specify specific HTTP methods if needed
    allow_headers=["*"],  # You can specify specific headers if needed
)

class CheckResult:
    def __init__(self, category: str, is_valid: bool, message: str):
        self.category = category
        self.is_valid = is_valid
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {
            "category": self.category,
            "is_valid": self.is_valid,
            "message": self.message
        }

# use perplexity ai
API_KEY = os.environ.get("PERPLEXITY_API_KEY")
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai/v1")
 

class ContentChecker:
    @staticmethod
    def default_checks():
        return [
            ContentChecker.check_pdf_validity,
            ContentChecker.check_pdf_pages,
            ContentChecker.check_pdf_text,
            ContentChecker.check_file_size,
            ContentChecker.check_fonts,
            ContentChecker.check_title_slide,
            ContentChecker.check_content_structure,
            ContentChecker.check_images,
            ContentChecker.check_audio,
            ContentChecker.check_ai_content_analysis
        ]

    @staticmethod
    def check_pdf(file_content: bytes, conference_topics: List[str], checks: List[Callable] = None, user_prompt: str = None) -> List[CheckResult]:
        results = []
        pdf_document = None

        try:
            pdf_document = pymupdf.open(stream=file_content, filetype="pdf")

            # Use provided checks if any, otherwise use default checks
            checks_to_run = checks if checks is not None else ContentChecker.default_checks()

            for check in checks_to_run:
                try:
                    sig = inspect.signature(check)
                    if len(sig.parameters) == 1:
                        result = check(pdf_document)
                    elif len(sig.parameters) == 2:
                        if 'topics' in sig.parameters:
                            result = check(pdf_document, conference_topics)
                        elif 'user_prompt' in sig.parameters:
                            result = check(pdf_document, user_prompt)
                        else:
                            result = check(pdf_document, file_content)
                    else:
                        raise ValueError(f"Unexpected number of parameters in {check.__name__}")

                    if isinstance(result, CheckResult):
                        results.append(result)
                    elif isinstance(result, List):
                        results.extend(result)
                except Exception as e:
                    logger.error(f"Error in check {check.__name__}: {str(e)}")
                    results.append(CheckResult(
                        category=f"Error in {check.__name__}",
                        is_valid=False,
                        message=f"Error: {str(e)}"
                    ))

            return results
        except Exception as e:
            logger.error(f"Error opening PDF: {str(e)}")
            return [CheckResult(
                category="PDF Opening Error",
                is_valid=False,
                message=f"Error opening PDF: {str(e)}"
            )]
        finally:
            if pdf_document:
                pdf_document.close()

    @staticmethod
    def check_pdf_validity(pdf_document: pymupdf.Document) -> CheckResult:
        return CheckResult(
            category="Slide Validity Check",
            is_valid=True,
            message="Valid PDF file"
        )

    @staticmethod
    def check_pdf_pages(pdf_document: pymupdf.Document) -> CheckResult:
        page_count = len(pdf_document)
        is_valid = 1 <= page_count <= 100
        
        message = f"Slide has {page_count} page{'s' if page_count != 1 else ''}"
        if not is_valid:
            message += ". This may be outside the expected range."
        
        return CheckResult(
            category="PDF Page Check",
            is_valid=is_valid,
            message=message
        )

    @staticmethod
    def check_pdf_text(pdf_document: pymupdf.Document) -> CheckResult:
        pdf_content = ""
        for page in pdf_document:
            pdf_content += page.get_text()

        has_text = len(pdf_content.strip()) > 0
        return CheckResult(
            category="Slide Content Check",
            is_valid=has_text,
            message="PDF contains text" if has_text else "PDF does not contain any text"
        )

    @staticmethod
    def check_file_size(pdf_document: pymupdf.Document, file_content: bytes) -> CheckResult:
        max_size_mb = 512  # 512MB limit as for perplexity ai
        file_size_bytes = len(file_content)
        file_size_mb = file_size_bytes / (1024 * 1024)
        is_valid = file_size_mb <= max_size_mb

        return CheckResult(
            category="Slide File Size",
            is_valid=is_valid,
            message=f"PDF size: {file_size_mb:.2f} MB" + (" (exceeds limit of %s MB)" % max_size_mb if not is_valid else "")
        )

    @staticmethod
    def check_fonts(pdf_document: pymupdf.Document) -> CheckResult:
        fonts = set(font[3] for page in pdf_document for font in page.get_fonts())
        
        return CheckResult(
            category="Fonts",
            is_valid=True,
            message=f"Fonts used: {', '.join(fonts) if fonts else 'None'}"
        )

    @staticmethod
    def check_title_slide(pdf_document: pymupdf.Document) -> CheckResult:
        def is_title_slide(page):
            # Extract text from the page
            text = page.get_text()
            
            # Check if the page contains very little text (typical for title slides)
            if len(text.strip()) < 50:
                return True
            
            # Check if the page contains common title slide keywords
            title_keywords = ["agenda", "outline", "contents", "introduction", "overview", "title", "presentation"]
            lower_text = text.lower()
            if any(keyword in lower_text for keyword in title_keywords):
                return True
            
            # Check if the page has large font text (typical for titles)
            blocks = page.get_text("blocks")
            if blocks:
                largest_font = max(block[5] for block in blocks)  # Font size is at index 5
                if largest_font > 24:  # Adjust this threshold as needed
                    return True
            
            return False

        if len(pdf_document) > 0:
            # Check the first page
            first_page = pdf_document[0]
            if is_title_slide(first_page):
                return CheckResult(
                    category="Title Slide Check",
                    is_valid=True,
                    message="The first page appears to be a title slide."
                )
            
            # Optionally, check the first few pages
            for i in range(min(3, len(pdf_document))):
                page = pdf_document[i]
                if is_title_slide(page):
                    return CheckResult(
                        category="Title Slide Check",
                        is_valid=True,
                        message=f"Page {i+1} appears to be a title slide."
                    )
            
            return CheckResult(
                category="Title Slide Check",
                is_valid=False,
                message="No clear title slide detected in the first few pages."
            )
        
        return CheckResult(
            category="Title Slide Check",
            is_valid=False,
            message="PDF has no pages"
        )

    @staticmethod
    def check_content_structure(pdf_document: pymupdf.Document) -> List[CheckResult]:
        total_words = 0
        total_images = 0
        total_bullet_points = 0
        MAX_BULLET_POINTS_PER_PAGE = 7  # Adjust this threshold as needed
        
        def count_bullet_points(text):
            bullet_patterns = [
                r'^\s*•',  # Bullet character
                r'^\s*-',  # Hyphen
                r'^\s*\*',  # Asterisk
                r'^\s*\d+\.',  # Numbered list (e.g., 1., 2., etc.)
                r'^\s*[a-zA-Z]\)',  # Alphabetic list (e.g., a), b), etc.)
            ]
            
            bullet_count = 0
            lines = text.split('\n')
            
            for line in lines:
                if any(re.match(pattern, line) for pattern in bullet_patterns):
                    bullet_count += 1
            
            return bullet_count
        
        pages_with_excess_bullets = []
        
        for page_num, page in enumerate(pdf_document, 1):
            text = page.get_text()
            words = text.split()
            total_words += len(words)
            total_images += len(page.get_images())
            
            bullet_count = count_bullet_points(text)
            total_bullet_points += bullet_count
            
            if bullet_count > MAX_BULLET_POINTS_PER_PAGE:
                pages_with_excess_bullets.append(page_num)
        
        avg_words_per_page = total_words / len(pdf_document) if len(pdf_document) > 0 else 0
        avg_images_per_page = total_images / len(pdf_document) if len(pdf_document) > 0 else 0
        avg_bullets_per_page = total_bullet_points / len(pdf_document) if len(pdf_document) > 0 else 0
        
        is_balanced = 50 <= avg_words_per_page <= 200 and 0.5 <= avg_images_per_page <= 5
        
        results = []
        
        results.append(CheckResult(
            category="Content Balance Check",
            is_valid=is_balanced,
            message=f"PDF contains a balanced mix of content (Avg. {avg_words_per_page:.1f} words and {avg_images_per_page:.1f} images per page)" if is_balanced else f"Content may not be well-balanced (Avg. {avg_words_per_page:.1f} words and {avg_images_per_page:.1f} images per page)"
        ))
        
        bullet_point_message = f"Average bullet points per page: {avg_bullets_per_page:.1f}"
        if pages_with_excess_bullets:
            bullet_point_message += f". Pages with excessive bullet points: {', '.join(map(str, pages_with_excess_bullets))}"
        
        results.append(CheckResult(
            category="Deterministic Bullet Point Check",
            is_valid=len(pages_with_excess_bullets) == 0,
            message=bullet_point_message
        ))
        
        return results

    @staticmethod
    def check_images(pdf_document: pymupdf.Document) -> CheckResult:
        total_images = sum(len(pdf_document.get_page_images(i)) for i in range(len(pdf_document)))
        has_images = total_images > 0
        
        return CheckResult(
            category="Images/Video Check",
            is_valid=has_images,
            message=f"PDF contains {total_images} image{'s' if total_images != 1 else ''}" if has_images else "No images detected in PDF"
        )

    @staticmethod
    def check_audio(pdf_document: pymupdf.Document) -> CheckResult:
        # Simple check for potential audio content
        for page in pdf_document:
            annots = page.annots()
            if any(annot.info.get('Subtype') in ['Sound', 'RichMedia'] for annot in annots):
                return CheckResult(
                    category="Audio Check",
                    is_valid=False,
                    message="Potential audio content detected in PDF"
                )
        
        return CheckResult(
            category="Audio Check",
            is_valid=True,
            message="No audio content detected in PDF"
        )
    

    @staticmethod
    def check_ai_content_analysis(pdf_document: pymupdf.Document, topics: List[str]) -> CheckResult:
        pdf_content = ""
        for page in pdf_document:
            pdf_content += page.get_text()

        messages = [
            {
                "role": "system",
                "content": "You are an artificial intelligence assistant for conference organizers",
            },
            {
                "role": "user",
                "content": f"""
                You are an AI conference organizer. Can you please check if the slides content fit the topics: {topics}
                Please reply in the json format 
                {{
                    "category": "AI Analysis",
                    "is_valid": boolean,
                    "message": str
                }}
                The content of the slides is as follows:
                {pdf_content[:4000]}  # Limit content to 4000 characters to avoid token limits
                """
            },
        ]

        client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

        try:
            response = client.chat.completions.create(
                model="mistral-7b-instruct",
                messages=messages,
            )

            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)

            # Create and return a CheckResult object
            return CheckResult(
                category="AI Content Analysis in " + result["category"],
                is_valid=result["is_valid"],
                message=result["message"]
            )
        except json.JSONDecodeError as e:
            return CheckResult(
                category="AI Content Analysis Error",
                is_valid=False,
                message=f"Failed to parse AI response: {str(e)}"
            )
        except Exception as e:
            return CheckResult(
                category="AI Analysis Error",
                is_valid=False,
                message=f"Error during AI analysis: {str(e)}"
            )
        
    @staticmethod
    def check_ai_custom(pdf_document: pymupdf.Document, user_prompt: str) -> CheckResult:
        pdf_content = ""
        for page in pdf_document:
            pdf_content += page.get_text()

        messages = [
            {
                "role": "system",
                "content": "You are an artificial intelligence assistant for conference organizers",
            },
            {
                "role": "user",
                "content": f"""
                You are an AI conference organizer. {user_prompt}
                Please reply in the json format 
                {{
                    "category": "AI Custom Analysis",
                    "is_valid": boolean,
                    "message": str
                }}
                The content of the slides is as follows:
                {pdf_content[:4000]}  # Limit content to 4000 characters to avoid token limits
                """
            },
        ]

        client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

        try:
            response = client.chat.completions.create(
                model="mistral-7b-instruct",
                messages=messages,
            )

            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)

            # Create and return a CheckResult object
            return CheckResult(
                category=result["category"],
                is_valid=result["is_valid"],
                message=result["message"]
            )
        except json.JSONDecodeError as e:
            return CheckResult(
                category="AI Content Analysis Error",
                is_valid=False,
                message=f"Failed to parse AI response: {str(e)}"
            )
        except Exception as e:
            return CheckResult(
                category="AI Analysis Error",
                is_valid=False,
                message=f"Error during AI analysis: {str(e)}"
            )
    
    @staticmethod
    def check_url(url: str) -> CheckResult:
        # Basic URL validation using regex
        url_pattern = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        is_valid_url = bool(url_pattern.match(url))
        
        if is_valid_url:
            try:
                result = urlparse(url)
                is_valid_url = all([result.scheme, result.netloc])
                
                # Check if it's a Google Slides URL
                if 'docs.google.com/presentation' in result.netloc + result.path:
                    return CheckResult(
                        category="URL Check",
                        is_valid=True,
                        message="Valid Google Slides URL"
                    )
                
                # Check if it's a Figma URL
                elif 'figma.com' in result.netloc:
                    return CheckResult(
                        category="URL Check",
                        is_valid=True,
                        message="Valid Figma URL"
                    )
                
                else:
                    return CheckResult(
                        category="URL Check",
                        is_valid=True,
                        message="Valid URL, but not Google Slides or Figma"
                    )
            except ValueError:
                is_valid_url = False
        
        return CheckResult(
            category="URL Check",
            is_valid=is_valid_url,
            message="Invalid URL"
        )

# Initialize the checks dictionary with the built-in checks
def initialize_checks():
    global checks
    checks = {}
    for name, func in vars(ContentChecker).items():
        if callable(func) and name.startswith("check_") and name not in ['check_pdf', 'check_url', 'check_ai_custom']:
            checks[name] = func

initialize_checks()

# Local storage
conferences = {}

# Pydantic models
class Conference(BaseModel):
    code: str
    topics: List[str] = []
    check_ids: List[str] = []

class Check(BaseModel):
    id: str
    function: str


class AddTopicRequest(BaseModel):
    code: str
    topic: str

class RemoveTopicRequest(BaseModel):
    code: str
    topic: str


# Dependency
def get_conference(code: str):
    if code not in conferences:
        logger.warning(f"Conference not found: {code}")
        raise HTTPException(status_code=404, detail="Conference not found")
    logger.info(f"Retrieved conference: {code}, Checks: {conferences[code].check_ids}")
    return conferences[code]

# API endpoints
@app.post("/api/add_conference")
async def api_add_conference(conference: Conference):
    if conference.code in conferences:
        raise HTTPException(status_code=400, detail="Conference already exists")

    # Assign all check IDs from the checks dictionary to the conference
    conference.check_ids = list(checks.keys())

    # Add default topic if no topics are provided
    if not conference.topics:
        conference.topics = ['AI', 'crypto']

    conferences[conference.code] = conference
    logger.info(f"Added conference: {conference.code}, Checks: {conference.check_ids}, Topics: {conference.topics}")
    return {"success": True, "message": "Conference added successfully with all available checks and default topic"}

@app.post("/api/remove_conference")
async def api_remove_conference(code: str):
    if code not in conferences:
        raise HTTPException(status_code=404, detail="Conference not found")
    del conferences[code]
    return {"success": True, "message": "Conference removed successfully"}

@app.get("/api/conferences")
async def api_get_conferences():
    return {"success": True, "conferences": list(conferences.keys())}

@app.post("/api/update_conference")
async def api_update_conference(code: str, conference: Conference):
    if code not in conferences:
        raise HTTPException(status_code=404, detail="Conference not found")
    conferences[code] = conference
    return {"success": True, "message": "Conference updated successfully"}

@app.get("/api/conference/{code}")
async def api_get_conference(conference: Conference = Depends(get_conference)):
    return {"success": True, "data": conference}

@app.get("/api/topics/{code}")
async def api_get_topics(code: str):
    conference = get_conference(code)
    return {"success": True, "topics": conference.topics}

@app.post("/api/add_topic")
async def api_add_topic(topic_request: AddTopicRequest):
    code = topic_request.code
    topic = topic_request.topic
    conference = get_conference(code)
    if topic not in conference.topics:
        conference.topics.append(topic)
        return {"success": True, "message": "Topic added successfully"}
    return {"success": False, "message": "Topic already exists"}

@app.post("/api/remove_topic")
async def api_remove_topic(remove_topic_request: RemoveTopicRequest):
    code = remove_topic_request.code
    topic = remove_topic_request.topic
    conference = get_conference(code)
    if topic in conference.topics:
        conference.topics.remove(topic)
        logger.info(f"Removed topic '{topic}' from conference '{code}'")
        return {"success": True, "message": "Topic removed successfully"}
    else:
        logger.warning(f"Topic '{topic}' not found in conference '{code}'")
        return {"success": False, "message": "Topic not found"}

@app.get("/api/checks/{conference_code}")
async def api_get_checks(conference_code: str):
    conference = get_conference(conference_code)
    if not conference:
        raise HTTPException(status_code=404, detail="Conference not found")

    # Fetch checks associated with the conference
    conference_checks = [check_id for check_id in conference.check_ids if check_id in checks]

    return {
        "success": True,
        "checks": conference_checks
    }

class AddCustomAICheckRequest(BaseModel):
    conference_code: str
    prompt: str
    
@app.post("/api/add_custom_ai_check")
async def api_add_custom_ai_check(request: AddCustomAICheckRequest):
    conference_code = request.conference_code
    prompt = request.prompt

    if conference_code not in conferences:
        raise HTTPException(status_code=404, detail="Conference not found")
    
    conference = conferences[conference_code]

    # Generate a unique check_id
    check_id = f"check_ai_custom_{uuid.uuid4().hex[:8]}"

    # Create the custom check function
    def custom_check(pdf_document: pymupdf.Document):
        return ContentChecker.check_ai_custom(pdf_document, user_prompt=prompt)

    # Set the __name__ attribute of the function for identification
    custom_check.__name__ = check_id

    # Add the function to the checks dictionary
    checks[check_id] = custom_check

    # Add the check_id to the conference's check_ids
    conference.check_ids.append(check_id)

    logger.info(f"Added custom AI check: {check_id} to conference: {conference_code}")

    return {"success": True, "message": "Custom AI check added successfully", "check_id": check_id}

class RemoveCheckRequest(BaseModel):
    check_id: str
    conference_code: str

# Updated API endpoint for removing a check
@app.post("/api/remove_check")
async def api_remove_check(remove_check_request: RemoveCheckRequest):
    check_id = remove_check_request.check_id
    conference_code = remove_check_request.conference_code

    if conference_code not in conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conference = conferences[conference_code]

    if check_id not in conference.check_ids:
        raise HTTPException(status_code=404, detail="Check not found in this conference")

    conference.check_ids.remove(check_id)
    logger.info(f"Removed check {check_id} from conference {conference_code}")

    return {"success": True, "message": f"Check {check_id} removed from conference {conference_code}"}

@app.post("/api/update_conference_checks")
async def api_update_conference_checks(code: str, check_ids: List[str]):
    conference = get_conference(code)
    conference.check_ids = check_ids
    return {"success": True, "message": "Conference checks updated successfully"}

@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), conference_code: str = Form(...)):
    try:
        logger.debug("Starting PDF upload process")
        content = await file.read()
        logger.info(f"File uploaded: {file.filename}, Content-Type: {file.content_type}, Size: {len(content)} bytes")

        # Check if the file is a PDF
        if not file.filename.lower().endswith('.pdf') or file.content_type != 'application/pdf':
            logger.warning(f"Invalid file type: {file.filename}, Content-Type: {file.content_type}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file type. Please upload a PDF file."}
            )

        conference_data = get_conference(conference_code)
        if not conference_data:
            return JSONResponse(
                status_code=404,
                content={"error": "Conference not found"}
            )

        checks_to_use = []
        for check_id in conference_data.check_ids:
            if check_id in checks:
                checks_to_use.append(checks[check_id])
                logger.debug(f"Using check: {check_id}")
            else:
                logger.warning(f"Check not found: {check_id}")

        logger.info(f"Using {len(checks_to_use)} checks: {[func.__name__ for func in checks_to_use]}")
        results = ContentChecker.check_pdf(content, conference_data.topics, checks_to_use)
        logger.debug(f"PDF checks completed. Results: {results}")
        return {"checks": [result.to_dict() for result in results]}
    except Exception as e:
        logger.exception(f"Error in upload_pdf: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )
    

@app.post("/api/check_url")
async def check_url(url: str = Form(...)):
    try:
        logger.debug(f"Checking URL: {url}")
        result = ContentChecker.check_url(url)
        logger.debug(f"URL check completed. Result: {result}")
        return {"checks": [result.to_dict()]}
    except Exception as e:
        logger.exception(f"Error in check_url: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"An unexpected error occurred: {str(e)}"}
        )

    
# Mount static files
app.mount("/", StaticFiles(directory="dist", html=True), name="static")

# Define the catch-all route after all other routes
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Exclude API routes
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404)
    return FileResponse("dist/index.html")






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
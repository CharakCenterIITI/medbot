from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn
import os
import pandas as pd
from dotenv import load_dotenv
import io
import json
import openai
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
# Load environment variables (e.g., OpenAI API Key)
load_dotenv()

# Set up the OpenAI API key


# Initialize FastAPI app
app = FastAPI(
    title="Dataset Analysis with GPT-3.5",
    version="1.0",
    description="API to handle dataset analysis using GPT-3.5."
)

# Route for handling CSV dataset queries
@app.post("/dataset/query")
async def analyze_dataset(file: UploadFile = File(...), query: str = Form(...)):
    try:
        # Ensure the uploaded file is a CSV
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="The uploaded file must be a CSV file.")
        
        # Read the CSV file content
        file_content = await file.read()
        dataset = pd.read_csv(io.BytesIO(file_content))

        dataset = dataset.head(100)

        # Limit the number of rows to avoid too much data
          # Limit to the first 10 rows (or you can adjust as needed)

        # Convert the dataset to a text-based format (tabular format or summary)
        dataset_text = dataset.to_string(index=False)  # Convert to a string without index

        # Prepare the prompt with dataset in text format and query
        prompt_text = f"Here is a portion of the dataset (only the first 10 rows):\n\n{dataset_text}\n\nNow, answer the following query: {query}"

        # Make the OpenAI API call for GPT-3.5 analysis
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}]
        )

        # Return the analysis result
        return {"analysis": response['choices'][0]['message']['content']}
    
    except openai.error.RateLimitError as e:
        # Handle rate limiting (if quota exceeded)
        raise HTTPException(status_code=429, detail="API rate limit exceeded. Please try again later.")
    
    except openai.error.APIError as e:
        # Handle general API errors
        raise HTTPException(status_code=500, detail=f"API error occurred: {str(e)}")
    
    except openai.error.AuthenticationError as e:
        # Handle authentication errors
        raise HTTPException(status_code=401, detail="Authentication failed. Please check your API key.")
    
    except Exception as e:
        # Catch other exceptions
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)





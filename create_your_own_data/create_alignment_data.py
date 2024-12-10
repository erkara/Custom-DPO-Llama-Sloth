import json
import csv
import os
from tqdm import tqdm
from docx import Document

from dotenv import load_dotenv
import pandas as pd

import torch
from langchain.schema import SystemMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI

import asyncio



"""Load environment variables and configure device.
CHANGE THE CONTENT OF all_keys.txt in the main repo!!!
"""
load_dotenv("../all_keys.txt")
hf_token = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    print(f"CUDA not found")



def extract_pages(directory_path, min_paragraph_length=20):
    """
    Extracts text from all .docx files in a directory and merges small paragraphs.
    """
    doc_texts = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".docx"):  # Process only .docx files
            file_path = os.path.join(directory_path, file_name)
            doc = Document(file_path)
            paragraphs = []

            buffer = ""  # Temporary buffer to merge small paragraphs
            for p in doc.paragraphs:
                text = p.text.strip()
                if len(text) < min_paragraph_length:
                    buffer += " " + text
                else:
                    if buffer:  # Append the buffer if it contains merged text
                        paragraphs.append(buffer.strip())
                        buffer = ""
                    paragraphs.append(text)
            if buffer:  # Add any remaining buffer
                paragraphs.append(buffer.strip())

            doc_texts[file_name] = "\n".join(paragraphs)  # Combine paragraphs into a single block
    return doc_texts



def chunk_text(directory_path, max_length=500, chunk_overlap=50):
    """
    Creates chunks for all .docx files in a directory.
    """
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=chunk_overlap)
    doc_texts = extract_pages(directory_path)

    for file_name, text in doc_texts.items():
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)  # Combine chunks from each document into a single list
    return all_chunks



async def generate_alignment_outputs(batch, client):
    """Processes a batch of requests asynchronously for alignment dataset.
        Carefully decide your goal here.
    """
    responses = []
    for chunk, template in batch:
        try:
            prompt = f"{template}\n\n{chunk}\n\nResponse:"
            
            # Generate informal output
            informal_messages = [
                SystemMessage(content="You are a helpful assistant. Always respond informally, like you're chatting with a friend."),
                HumanMessage(content=prompt),
            ]
            informal_response = client(informal_messages)
            
            # Generate formal output
            formal_messages = [
                SystemMessage(content="You are a professional assistant. Always respond formally and concisely."),
                HumanMessage(content=prompt),
            ]
            formal_response = client(formal_messages)
            
            # Collect both responses
            responses.append({
                "Instruction": template,
                "Prompt": chunk,
                "Output_Informal": informal_response.content.strip(),
                "Output_Formal": formal_response.content.strip(),
                "Preferred": "Output_Informal",  # Default preference; adjust if needed
            })
        except Exception as e:
            print(f"Error processing request: {e}")
            responses.append({
                "Instruction": template,
                "Prompt": chunk,
                "Output_Informal": "Error",
                "Output_Formal": "Error",
                "Preferred": "Error",
            })
    return responses


async def generate_alignment_pairs(client, chunks, output_file="alignment_dataset.csv", batch_size=10):
    """Generate alignment pairs with dynamic saving."""
    data = []
    batch = []
    total_tasks = len(chunks) * len(instruction_templates)

    with tqdm(total=total_tasks, desc="Generating Alignment Pairs", unit="task") as pbar, open(
        output_file, mode="a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=["Instruction", "Prompt", "Output_Informal", "Output_Formal", "Preferred"])
        
        # Write the header if the file is new
        if os.path.getsize(output_file) == 0:
            writer.writeheader()

        for chunk in chunks:
            for template in instruction_templates:
                batch.append((chunk, template))  # Add task to batch
                if len(batch) == batch_size:  # If batch is full, process it
                    batch_results = await generate_alignment_outputs(batch, client)
                    data.extend(batch_results)
                    writer.writerows(batch_results)  # Save dynamically
                    batch.clear() 
                    pbar.update(batch_size)

        # Process any remaining requests in the batch
        if batch:
            batch_results = await generate_alignment_outputs(batch, client)
            data.extend(batch_results)
            writer.writerows(batch_results)  # Save dynamically
            pbar.update(len(batch))

    return data


# Main Script
if __name__ == "__main__":

    # Initialize OpenAI client
    client = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Directory and settings
    directory_path = "data"
    output_csv = "alignment_dataset.csv"
    max_length = 1000
    chunk_overlap = 100

    # Generate combined chunks from all documents
    chunks = chunk_text(directory_path, max_length=max_length, chunk_overlap=chunk_overlap)

    # Instruction templates--> make sure to modify it.
    instruction_templates = [
        "Answer a question based on the following content.",
        "What is a relevant question that could be asked about the following content?",
        "Explain the key information in the following content.",
        "What facts can you derive from the following content?",
        "Provide a detailed explanation based on the following content.",
        "Summarize the following content as it relates to answering questions.",
        "What key insights can be drawn from the following content?",
        "Using the following content, provide relevant details to answer a question.",
        "What specific knowledge does the following content provide?",
        "Generate a plausible question that the following content can answer.",
    ]

    # Run the asynchronous batch processing with dynamic saving
    data = asyncio.run(
        generate_alignment_pairs(client, chunks, output_file=output_csv, batch_size=5)
    )

    print("Alignment dataset creation complete!")

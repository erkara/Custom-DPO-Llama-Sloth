# Instruction Dataset Generation
This script is designed to create an **instruction-prompt-output dataset** for fine-tuning or alignment tasks. It processes `.docx` files, chunks the text, applies pre-defined instruction templates, and uses a language model to generate responses. Before starting off, go to `all_keys.txt` and enter your OpenAI and Hugging Face keys there.

## Frame the Problem: 
The goal is to take long, rich documents and turn them into a dataset of instructions and responses. Here's the process:

1. **Start with Source Texts**: I created a niche dataset all about my tiny home town (not city), [Honaz](https://en.wikipedia.org/wiki/Honaz) in Turkey. This dataset was built entirely from three `Turkish` articles in the [DergiPark](https://dergipark.org.tr/) system. I considered the following articles:

[Haytoğlu E. Denizli-Honaz’a yapılan mübadele göçü](https://acikerisim.deu.edu.tr/xmlui/handle/20.500.12397/4848). *Çağdaş Türkiye Tarihi Araştırmaları Dergisi*, 2006; 5(12): 47–65.

[Aydın M. 19. Yüzyıldan 20. Yüzyıla Honaz](https://dergipark.org.tr/tr/pub/pausbed/issue/34751/384343). *Pamukkale Üniversitesi Sosyal Bilimler Enstitüsü Dergisi*. 2016(25):199-227.

[Büyükoğlan F. Honaz Dagi ve Çevresinin Bitki Örtusu](https://dergipark.org.tr/tr/pub/kefdergi/issue/49063/625995). *Kastamonu Education Journal*. 2010;18(2):631-52.


The idea here is simple: take rich, detailed information about a small town in Turkish (because that's the language of the original sources), and see if a model can actually learn from it. Why bother creating such a niche dataset in Turkish? Well, fine-tuning models on localized, specific datasets lets us test if they can adapt to truly unique and specialized topics. This is good because we can easily assess if the model really learns something. If you have a look at the Jupyter files in the main repo, you will see that before fine-tuning, LLM simply hallucinates. 

2. **Break the Text into Chunks**:  Large documents are split into smaller, manageable parts while keeping the meaning intact.

3. **Apply Instruction Templates**: Each chunk is paired with multiple instructions, like "Summarize this" or "Extract key phrases."

4. **Generate Responses**:A language model processes the instructions and generates responses.

5. **Save Everything**: The results are saved incrementally into a CSV file so nothing gets lost with some additional considerations in mind. 

---

## Step 2: Code Breakdown

`extract_pages`: This function reads `.docx` files and pulls out text, ensuring that small paragraphs are merged into meaningful sections. I deliberately chose to work with docx files since we can easily add/remove stuff and restructure it and we dont need to think about hasty OCR pipeline for PDFs.

`chunk_text`: Splits the extracted text into smaller pieces, each around 500 tokens long. Overlaps are added between chunks so that the context isn't lost.

`generate_batch_outputs` This is the heart of the script. It:
- Takes a batch of chunks and instructions.
- Asks the language model (via an API) to generate responses for each instruction.
- Uses Python's `asyncio` to process multiple requests at once, avoiding API rate limits and saving time.
If any requests fail, they’re logged so you can retry later without losing progress.

I specifically used `gpt-4o-mini`to generate responses. I know some folks do not want to put their credit card info there but to give you an idea, it costs less than $0.5 to generate 1000 instruction-response pair. Anyways, for any paid service like OpenAI APIs, using `asyncio` is a good practice because all of these services operate under some sort of hourly/daily requests/token limits. We need to be strategical about how we pin the service. You can customize the code to use open-source models as well.


`generate_instruction_pairs`: This orchestrates everything:
- Combines all chunks and instructions into batches.
- Calls `generate_batch_outputs` to process them asynchronously.
- Saves the outputs into a CSV file as they’re generated. No progress is lost, even if the script is interrupted.

---

# Alignment Data Generation

This script helps create an **alignment dataset** to fine-tune models for preference-aware tasks like **DPO (Direct Preference Optimization)** or **RLHF (Reinforcement Learning with Human Feedback)**. The dataset consists of instruction-prompt-output pairs where outputs are generated in two styles (e.g., formal and informal) to align the model's behavior with user preferences. This is something you decide based on your task.

---


1. **Source Preparation, Extract and Chunk Text***:  this part is exactly the same above.
2. **Generate Paired Outputs**:Each text chunk is paired with multiple instructions. For every chunk-instruction pair:
     - **Informal Output**: A chat-like, casual response.
     - **Formal Output**: A polished, professional response.
   - Both outputs are stored with a default preference (e.g., informal by default). You can easily customize this part based on your preference.
 Rest of the code is pretty much the same as above.  If you have energy, you can check some of the pairs but GPT4 does a pretty good job.


I published both datasets in [my Hugging Face space](https://huggingface.co/erdi28), feel free to use it. 



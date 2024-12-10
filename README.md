## Llama-3.2 Fine-Tuning and Alignment Using Custom Dataset

This repository walks you through the process of fine-tuning and aligning **Llama-3.2-1B-Instruct** using a custom dataset I created about my tiny hometown, [Honaz](https://en.wikipedia.org/wiki/Honaz), in Turkey. The goal is to train a model that can genuinely absorb the knowledge of a niche topic. 

### The Dataset: Honaz (my hometown)

I built a unique dataset about **Honaz**, a small town with a rich history and natural beauty, using three academic articles in Turkish from the [DergiPark](https://dergipark.org.tr/) platform.

1. [Haytoğlu E. Denizli-Honaz’a yapılan mübadele göçü](https://acikerisim.deu.edu.tr/xmlui/handle/20.500.12397/4848). *Çağdaş Türkiye Tarihi Araştırmaları Dergisi*, 2006; 5(12): 47–65.
2. [Aydın M. 19. Yüzyıldan 20. Yüzyıla Honaz](https://dergipark.org.tr/tr/pub/pausbed/issue/34751/384343). *Pamukkale Üniversitesi Sosyal Bilimler Enstitüsü Dergisi*, 2016(25): 199–227.
3. [Büyükoğlan F. Honaz Dagi ve Çevresinin Bitki Örtusu](https://dergipark.org.tr/tr/pub/kefdergi/issue/49063/625995). *Kastamonu Education Journal*, 2010; 18(2): 631–52.

The idea is simple: take detailed, localized information about a small town (originally in Turkish), translate it into English, and test if a model can genuinely learn from such niche data. Fine-tuning on specific topics like this helps assess how well models adapt to unique, specialized content. When you look at the first notebook, you will see that before fine-tuning, the model simply hallucinate when asked about Honaz. 


### What's in the Repo

**Data Generation**:  The scripts for creating instruction and alignment datasets are in the `create_your_own_data` folder:
     - `create_instruction_data.py`: Generates instruction datasets with prompts and responses.
     - `create_alignment_data.py`: Builds datasets for comparing formal and informal responses.
   - These datasets are already uploaded to **HuggingFace**, so you can skip the generation step if you wish. Note: I have not added *readme* to these repos yet, I will do it soon.

**Fine-Tune the Model**: Use `llama_finetune_honaz.ipynb` to fine-tune Llama-3.2-1B-Instruct on the Honaz dataset using the [Unsloth](https://github.com/unslothai/unsloth) . It is pretty fast compared to native transformers. In the `evaluation` folder, you’ll find 9 custom questions with answers to test how well the model has learned about Honaz. These questions help assess the model’s knowledge, for sure 1000x better approach then using BLEU score.

**Align the Model**: Use `dpo_llama_alignment.ipynb` to apply Direct Preference Optimization (DPO) and align the model to informal responses.

Both notebooks are fairly detailed, outlining all the keys details in pre/post training phases. 
`NOTE`: Make sure to update the required keys in "all_keys.txt" to enable access to OpenAI, Hugging Face, and Weights & Biases.

## Why This Repo Matters

It’s common to rely on publicly available datasets, use default training parameters, and quickly move forward. While this is convenient, I believe it’s a good practice to create your own dataset at least once in your lifetime. :) If you’re working as an LLM practitioner, this is something you’re likely to encounter sooner or later. This repo walks you through the entire process—starting from scratch, creating datasets, fine-tuning a model, aligning it to user preferences, and evaluating its performance—all in one place.

Hope you enjoy!
Erdi Kara: erdi_kara88@hotmail.com

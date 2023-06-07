cd "G:\My Drive\GPT Testing\Source"

call "C:\Users\evanh\Documents\Python Environments\OpenAI Venv\Scripts\activate.bat"
set PYTHONUNBUFFERED=1
set OPENAI_API_KEY=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
python -m main --models "text-davinci-002" --datasets "23step" --modalities "suppressed_cot" --num_samples=500 --use_simple_prompt True --extraction_type "in-brackets" --continuation True --wait_time 0 --max_tokens 2000

PAUSE
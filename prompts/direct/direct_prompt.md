# Cell 5: Direct prompt template
DIRECT_PROMPT = """You are an expert English-Chinese bilingual speaker. Given an English dialogue, please write a concise Simplified Chinese summary. Output only the summary, without titles or markdown.

Dialogue:
{dialogue}

Chinese summary:"""

print(DIRECT_PROMPT)
# Agentic-Biomedical-Retrieval-System

Topic : Execution for the Agent
# from the folder where app.py lives
-> uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# To know the health
-> curl -s http://127.0.0.1:8000/health | python -m json.tool
# To train the model initially with the cleaned notes (If MIMIC notes are not cleaned refer to steps mentioned ahead)
-> curl -X POST "http://127.0.0.1:8000/load_csv?text_col=TEXT&context_col=CONTEXT&id_col=HADM_ID&chunk_size=1200&chunk_overlap=200" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/workspace/final_notes_clean.csv"
# To run a query for retrieval (Make sure there is no gap between the three lines)
-> curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"red skin after antibiotics","top_k":16,"final_k":5,"reflect":true}' | python -m json.tool (edited) 





9:48
Topic : MIMIC Notes Cleaning
# To prepare the mimic notes clean them first (careful with the file locations as mentioned below)
-> python repair_csv_quotes.py /workspace/JDRE-agent/Agent_Dhruv/final_notes.csv /workspace/JDRE-agent/Agent_Dhruv/final_notes_clean.csv
9:50
Prevention : Hardcoded the OpenAI API key in the file rag_config.py
-> OPENAI_API_KEY = "REPLACE_WITH_YOUR_OPENAI_KEY" (edited) 

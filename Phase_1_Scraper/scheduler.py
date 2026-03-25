import schedule
import time
import subprocess
import datetime
import os
import shutil

def run_pipeline():
    print(f"\\n--- [{datetime.datetime.now()}] Starting Automated Data Refresh Pipeline ---")
    
    # Step 1: Run Scraper
    print("[1/2] Running scraper.py to fetch latest courses...")
    scraper_result = subprocess.run(["python", "scraper.py"])
    if scraper_result.returncode != 0:
        print("❌ Scraper encountered an error. Aborting pipeline refresh.")
        return
        
    print("✅ Scraper finished successfully.")

    # Step 2: Run Vector Ingestion
    print("[2/2] Rebuilding Vector Database...")
    if os.path.exists("../Phase_2_RAG/chroma_db"):
        try:
            shutil.rmtree("../Phase_2_RAG/chroma_db")
            print("Old chroma_db removed for clean ingestion.")
        except Exception as e:
            print(f"⚠️ Could not remove old vector db: {e}")
            
    vector_result = subprocess.run(["python", "vector_store.py"], cwd="../Phase_2_RAG")
    if vector_result.returncode != 0:
        print("❌ Vector Store ingestion encountered an error.")
        return
        
    print(f"✅ [{datetime.datetime.now()}] Automated Data Refresh Pipeline Completed Successfully!\\n")

# Setup Schedule Config
# Run the pipeline every day at Midnight (00:00)
schedule.every().day.at("00:00").do(run_pipeline)

# For testing functionality, you can uncomment the line below to run every 1 minute:
# schedule.every(1).minutes.do(run_pipeline)

print(f"[{datetime.datetime.now()}] Karan beyond UPSC - Data Refresh Scheduler Started.")
print("The data ingest pipeline will automatically run everyday at Midnight (00:00).")
print("Keeping process alive. Waiting for scheduled jobs...")

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60) # check the schedule every minute

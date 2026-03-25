import asyncio
import json
from playwright.async_api import async_playwright

async def scroll_to_bottom(page):
    await page.evaluate("""
        async () => {
            await new Promise((resolve) => {
                let totalHeight = 0;
                let distance = 500;
                let timer = setInterval(() => {
                    let scrollHeight = document.body.scrollHeight;
                    window.scrollBy(0, distance);
                    totalHeight += distance;
                    if(totalHeight >= scrollHeight - window.innerHeight){
                        clearInterval(timer);
                        resolve();
                    }
                }, 200);
            });
        }
    """)

async def scrape_pw_batches(url="https://www.pw.live/upsc"):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        print(f"Navigating to {url}...")
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            print("Scrolling down the page to load all dynamical content...")
            await scroll_to_bottom(page)
            await page.wait_for_timeout(3000)
            
            print("Extracting page content...")
            body_text = await page.inner_text("body")
            
            # Clean text to remove header (other exams like JEE) and footer
            start_marker = "UPSC Coaching"
            end_marker = "Join 15 Million students on the app today!"
            
            if start_marker in body_text:
                body_text = "UPSC\\n" + body_text[body_text.index(start_marker):]
            if end_marker in body_text:
                body_text = body_text[:body_text.index(end_marker)]
                
            # Some cleaning to remove heavy empty lines
            cleaned_text = "\\n".join([line.strip() for line in body_text.split("\\n") if line.strip()])
            
            data = {
                "source": url,
                "content": cleaned_text
            }
            
            with open("courses_data.json", "w", encoding="utf-8") as f:
                json.dump([data], f, indent=4, ensure_ascii=False)
                
            print("Successfully saved scraped content to courses_data.json")
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(scrape_pw_batches())

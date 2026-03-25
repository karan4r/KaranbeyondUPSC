import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print("Navigating to https://www.pw.live/study-v2/batches")
            response = await page.goto('https://www.pw.live/study-v2/batches', timeout=60000)
            if response:
                print(f"Status: {response.status}")
            else:
                print("No response object returned.")
            
            print("Waiting for page to load...")
            await page.wait_for_timeout(5000) # Wait for React/Angular to load content
            
            content = await page.content()
            with open("pw_page.html", "w", encoding="utf-8") as f:
                f.write(content)
            print("Saved page content to pw_page.html")
            
            await page.screenshot(path="pw_page.png", full_page=False)
            print("Saved screenshot to pw_page.png")
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

def scrape_match_stats():
    # URL k analýze
    url = "https://www.livesport.cz/zapas/fotbal/arsenal-hA1Zm19f/manchester-utd-ppjDR086/prehled/stats/0/?mid=YqgQO69M"
    
    # Nastavení Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Spustit v headless módu (bez viditelného okna)
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Inicializace webdriveru
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print("Otevírám URL...")
        driver.get(url)
        
        # Počkat na načtení stránky
        print("Čekám na načtení stránky...")
        time.sleep(3)
        
        # Hledání elementu se statistikami - budeme hledat různé možné selektory
        stats_selectors = [
            ".statsContent",
            ".statisticsContent", 
            ".matchStats",
            "[data-testid='statistics']",
            ".statsTable",
            ".statisticsTable",
            ".detailTabContent",
            ".tabContent"
        ]
        
        stats_found = False
        stats_content = ""
        
        for selector in stats_selectors:
            try:
                print(f"Hledám statistiky pomocí selektoru: {selector}")
                wait = WebDriverWait(driver, 10)
                stats_element = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                stats_content = stats_element.text
                if stats_content.strip():
                    stats_found = True
                    print(f"Našel jsem statistiky pomocí selektoru: {selector}")
                    break
            except Exception as e:
                print(f"Selektor {selector} nebyl nalezen: {e}")
                continue
        
        # Pokud nenajdeme specifický kontejner, zkusíme najít jakékoliv statistiky na stránce
        if not stats_found:
            print("Hledám obecné statistiky na stránce...")
            try:
                # Hledání všech elementů, které mohou obsahovat statistiky
                possible_stats = driver.find_elements(By.CSS_SELECTOR, "div[class*='stat'], div[class*='Stat'], td[class*='stat'], span[class*='stat']")
                if possible_stats:
                    stats_content = "\n".join([elem.text for elem in possible_stats if elem.text.strip()])
                    stats_found = True
            except Exception as e:
                print(f"Chyba při hledání obecných statistik: {e}")
        
        # Pokud stále nic nenalezeno, zkusíme získat celý obsah stránky
        if not stats_found:
            print("Získávám celý textový obsah stránky...")
            stats_content = driver.find_element(By.TAG_NAME, "body").text
        
        # Výpis statistik
        print("\n" + "="*50)
        print("STATISTIKY ZÁPASU:")
        print("="*50)
        print(stats_content)
        print("="*50)
        
        # Uložení do souboru
        with open("match_stats.txt", "w", encoding="utf-8") as f:
            f.write(stats_content)
        print("\nStatistiky byly uloženy do souboru 'match_stats.txt'")
        
    except Exception as e:
        print(f"Došlo k chybě: {e}")
        
    finally:
        driver.quit()
        print("Prohlížeč byl uzavřen.")

if __name__ == "__main__":
    scrape_match_stats()

"""
Модуль поиска в интернете через DuckDuckGo
"""

from duckduckgo_search import DDGS

class InternetSearch:
    """Поиск в интернете через DuckDuckGo"""
    
    def __init__(self):
        pass
    
    def search(self, query, max_results=3):
        """Поиск информации в интернете"""
        try:
            print(f"Поиск в интернете: {query}")
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            # Извлечение текста
            snippets = []
            for i, r in enumerate(results[:max_results], 1):
                title = r.get('title', 'Без названия')
                body = r.get('body', '')
                snippet = f"{i}. {title}\n{body}"
                snippets.append(snippet)
            
            if not snippets:
                return "Ничего не найдено"
            
            return "\n\n".join(snippets)
        
        except Exception as e:
            return f"Ошибка поиска: {str(e)}"
    
    def search_wikipedia(self, query, lang="ru"):
        """Поиск в Википедии"""
        try:
            import wikipedia
            
            wikipedia.set_lang(lang)
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return "Ничего не найдено в Википедии"
            
            snippets = []
            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    snippet = f"Wikipedia: {title}\n{page.summary[:500]}"
                    snippets.append(snippet)
                except:
                    continue
            
            if not snippets:
                return "Страницы не найдены"
            
            return "\n\n".join(snippets)
        
        except ImportError:
            return "Установите: pip install wikipedia"
        except Exception as e:
            return f"Ошибка Wikipedia: {str(e)}"

if __name__ == "__main__":
    # Тест
    search = InternetSearch()
    
    print("Тест 1: Общий поиск")
    result = search.search("Гоголь Мёртвые души сюжет")
    print(result[:500])
    print("\n" + "="*60 + "\n")
    
    print("Тест 2: Википедия")
    result = search.search_wikipedia("Гоголь Николай")
    print(result[:500])

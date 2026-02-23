import wikipediaapi
print("Import success")
try:
    wiki = wikipediaapi.Wikipedia('FakenewsPortfolio/1.0 (contact@example.com)', 'en')
    page = wiki.page('Python (programming language)')
    print(f"Page exists: {page.exists()}")
    print(f"Summary: {page.summary[:50]}...")
except Exception as e:
    print(f"Error: {e}")

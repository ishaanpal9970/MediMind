"""Small helper to list providers using the project's ORM.
Run inside the project's virtualenv:

  (venv) $ python scripts/list_providers.py

This script is intentionally read-only.
"""

from src.medimind_backend import app, Provider

if __name__ == '__main__':
    with app.app_context():
        providers = Provider.objects()
        print('Providers in database:')
        if not providers:
            print('  (none)')
        for p in providers:
            print(f'ID: {p.id}, Name: {p.name}, Type: {p.type}, Verified: {p.is_verified}')

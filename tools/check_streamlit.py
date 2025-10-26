import asyncio
import importlib

print('python:', __import__('sys').executable)

for pkg in ('streamlit','tornado'):
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', None) or getattr(m, 'version', None) or 'unknown'
        print(f'{pkg}: {v}')
    except Exception as e:
        print(f'{pkg}: import error: {e}')

try:
    loop = asyncio.get_running_loop()
    print('asyncio.get_running_loop(): running loop found')
except RuntimeError as e:
    print('asyncio.get_running_loop(): RuntimeError (no running loop)')
except Exception as e:
    print('asyncio.get_running_loop():', type(e).__name__, e)

print('Check complete')

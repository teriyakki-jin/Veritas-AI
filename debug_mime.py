import mimetypes
import os

print(f"Initial mapping for .js: {mimetypes.guess_type('app.js')[0]}")

mimetypes.init()
print(f"Post-init mapping for .js: {mimetypes.guess_type('app.js')[0]}")

mimetypes.add_type("application/javascript", ".js")
print(f"Post-fix mapping for .js: {mimetypes.guess_type('app.js')[0]}")

# Check registry if possible (windows specific)
try:
    import winreg
    key = winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, ".js")
    val, type = winreg.QueryValueEx(key, "Content Type")
    print(f"Registry Content Type for .js: {val}")
except Exception as e:
    print(f"Could not read registry: {e}")

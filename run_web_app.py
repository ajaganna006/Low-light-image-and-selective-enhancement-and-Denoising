"""
Quick launcher for the web application
"""

import subprocess
import sys
import os

def install_flask():
    """Install Flask if not available"""
    try:
        import flask
        print("✓ Flask is already installed")
    except ImportError:
        print("Installing Flask...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
        print("✓ Flask installed successfully")

def main():
    print("Low Light Image Enhancement - Web App")
    print("=" * 50)
    
    # Install Flask if needed
    install_flask()
    
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    print("\nStarting web application...")
    print("The app will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run the web app
    try:
        from web_app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\nWeb application stopped.")
    except Exception as e:
        print(f"Error starting web application: {e}")

if __name__ == '__main__':
    main()

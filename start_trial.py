"""
Trial Launcher - Easy way to start the trial
"""

import subprocess
import sys
import os

def main():
    print("🎯 Low Light Image Enhancement - TRIAL LAUNCHER")
    print("=" * 50)
    print("Welcome to the trial version!")
    print("This will demonstrate the key features of our enhancement system.")
    print()
    
    print("Choose your trial option:")
    print("1. 🎨 Demo Trial (Recommended) - Creates sample images")
    print("2. 🖼️  Custom Image Trial - Use your own image")
    print("3. 🌐 Web Interface Trial - Full web application")
    print("4. 🚀 Quick Demo - Full feature demonstration")
    print()
    
    while True:
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\n🎨 Starting Demo Trial...")
            subprocess.run([sys.executable, "trial.py"])
            break
        elif choice == '2':
            image_path = input("Enter path to your image: ").strip()
            if image_path:
                print(f"\n🖼️  Starting Custom Image Trial with: {image_path}")
                subprocess.run([sys.executable, "trial.py", "--image", image_path])
            else:
                print("❌ No image path provided. Starting demo trial instead.")
                subprocess.run([sys.executable, "trial.py"])
            break
        elif choice == '3':
            print("\n🌐 Starting Web Interface Trial...")
            print("The web interface will open at: http://localhost:5000")
            print("Press Ctrl+C to stop the server")
            subprocess.run([sys.executable, "run_web_app.py"])
            break
        elif choice == '4':
            print("\n🚀 Starting Quick Demo...")
            subprocess.run([sys.executable, "quick_demo.py"])
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    print("\n" + "=" * 50)
    print("🎉 Trial completed!")
    print("Thank you for trying our Low Light Image Enhancement system!")
    print("=" * 50)

if __name__ == '__main__':
    main()

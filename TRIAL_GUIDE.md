# 🎯 Trial Guide - Low Light Image Enhancement

## 🚀 **How to Start a Trial**

### **Option 1: Easy Trial Launcher (Recommended)**
```bash
python start_trial.py
```
- Interactive menu with 4 trial options
- Choose what you want to test
- Guided experience

### **Option 2: Direct Trial**
```bash
python trial.py
```
- Runs demo trial immediately
- Creates sample images and shows enhancement
- No setup required

### **Option 3: Custom Image Trial**
```bash
python trial.py --image your_image.jpg
```
- Test with your own image
- Shows before/after comparison
- Saves enhanced results

---

## 🎨 **Trial Options Explained**

### **1. Demo Trial** 
- **What it does**: Creates a sample image, makes it dark, then enhances it
- **Shows**: 5 different enhancement methods side-by-side
- **Files created**: 
  - `trial_original.png` - Original sample image
  - `trial_dark.png` - Dark version
  - `trial_enhanced.png` - Best enhanced result
  - `trial_results.png` - Comparison of all methods

### **2. Custom Image Trial**
- **What it does**: Takes your image, creates a dark version, enhances it
- **Shows**: Before/after comparison with your image
- **Files created**: 
  - `your_image_trial_dark.png` - Dark version of your image
  - `your_image_trial_enhanced.png` - Enhanced version
  - `your_image_trial_comparison.png` - Side-by-side comparison

### **3. Web Interface Trial**
- **What it does**: Opens a beautiful web interface
- **Shows**: Drag & drop image enhancement
- **Features**: 
  - Multiple enhancement methods
  - Real-time processing
  - Download results
  - Professional interface

### **4. Quick Demo Trial**
- **What it does**: Full feature demonstration
- **Shows**: All capabilities of the system
- **Features**: 
  - Synthetic dataset creation
  - Multiple enhancement techniques
  - Comprehensive visualizations

---

## 🎯 **What You'll See in the Trial**

### **Enhancement Methods Demonstrated**
1. **⚡ Gamma Correction** - Adjusts brightness curves
2. **📊 Histogram Equalization** - Improves contrast distribution  
3. **🎯 CLAHE** - Adaptive histogram equalization
4. **💡 Brightness/Contrast** - Direct adjustment
5. **🚀 Comprehensive** - Combines multiple techniques (best results)

### **Visual Results**
- **Before/After Comparisons** - Clear visual improvements
- **Multiple Methods** - See different enhancement approaches
- **Professional Quality** - High-resolution output images
- **Instant Processing** - Real-time enhancement

---

## 📁 **Trial Files Generated**

After running any trial, you'll get:

```
📁 Trial Results/
├── trial_original.png          # Original sample image
├── trial_dark.png             # Low light version
├── trial_enhanced.png         # Enhanced result
├── trial_results.png          # Comparison visualization
└── [your_image]_trial_*.png   # Custom image results (if used)
```

---

## 🎉 **Trial Benefits**

### **✅ See Immediate Results**
- No complex setup required
- Works with any image format
- Instant visual feedback

### **✅ Test Multiple Methods**
- Compare different enhancement techniques
- See which works best for your images
- Understand the differences

### **✅ Professional Quality**
- High-resolution output
- Multiple format support
- Beautiful visualizations

### **✅ Easy to Use**
- Simple commands
- Clear instructions
- Guided experience

---

## 🚀 **After the Trial**

### **Ready for More? Try the Full Version:**

```bash
# Full demo with all features
python quick_demo.py

# Web interface for regular use
python run_web_app.py

# Train your own deep learning model
python minimal_train.py --epochs 10

# Comprehensive demonstration
python demo.py
```

### **What's Available in Full Version:**
- **Advanced Deep Learning Models** - Train custom enhancement models
- **Batch Processing** - Enhance multiple images at once
- **Web Interface** - Professional drag & drop interface
- **Custom Training** - Use your own image datasets
- **Advanced Algorithms** - More sophisticated enhancement methods

---

## 💡 **Trial Tips**

### **For Best Results:**
1. **Use good quality images** - Higher resolution = better results
2. **Try different methods** - Each works better for different image types
3. **Compare side-by-side** - Look at the generated comparison images
4. **Test with your own images** - Use the custom image trial option

### **Supported Image Formats:**
- PNG, JPG, JPEG, BMP, TIFF
- Any size (automatically resized if too large)
- Color and grayscale images

---

## 🎯 **Quick Start Commands**

```bash
# Start with this (easiest)
python start_trial.py

# Or go directly to demo
python trial.py

# Test with your image
python trial.py --image photo.jpg

# Try the web interface
python run_web_app.py
```

**🎉 Enjoy your trial experience!**

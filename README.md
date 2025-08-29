第一次需要创建虚拟空间：  
python -m venv pixvenv  

pixvenv\Scripts\activate  

安装模块：  
pip install pillow numpy  

以后就只需要：  
pixvenv\Scripts\python.exe run.py big_pixel.png restored.png --scale 15

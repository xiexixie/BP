# BP
A simple Back propagation neural network in c++  
## 网络结构
本次项目的BP神经网络结构采用最简单的三层结构：  
        
        输入层:1 
        隐藏层:1
        输出层:1    
## 项目结构
```
BP:     
├─CMakeLists.txt  
├─LICENSE  
├─main.cpp  
├─README.md  
├─test_data.txt  
├─train_data.txt  
├─include  
│    ├─net.h  
│    ├─timer.h  
│    └─tools.h  
├─old  
│    ├─bp.cpp  
│    └─bp.exe  
└─src  
     ├─net.cpp  
     └─tools.cpp  
```  
## 测试实例
`注：本项目没有模型保存功能，每次训练均为重新开始，结果可能不同`    

训练数据：     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`train_data.txt`  
测试数据：     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`test_data.txt`  
输入层节点:    &nbsp;&nbsp;&nbsp;&nbsp;`IN_NODE 2`      
隐藏层节点:    &nbsp;&nbsp;&nbsp;&nbsp;`HIDE_NODE 4`  
输出层节点:    &nbsp;&nbsp;&nbsp;&nbsp;`OUT_NODE 1`   
学习率:        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`rate 0.8`  
最大误差:      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`theresoid 1e-4`  
最大迭代次数:  &nbsp;`max_times 1e6`  
<br>
训练数据如下：  
      
      0 0 0
      0 1 1
      1 0 1
      1 1 0
      0.8 0.8 0
      0.6 0.6 0
      0.4 0.4 0
      0.2 0.2 0
      1.0 0.8 1
      1.0 0.6 1
      1.0 0.4 1
      1.0 0.2 1
      0.8 0.6 1
      0.6 0.4 1
      0.4 0.2 1
      0.2 0 1
      0.999 0.666 1
      0.666 0.333 1
      0.333 0 1
      0.8 0.4 1
      0.4 0 1
      0 0.123 1
      0.12 0.23 1
      0.23 0.34 1
      0.34 0.45 1
      0.45 0.56 1
      0.56 0.67 1
      0.67 0.78 1
      0.78 0.89 1
      0.89 0.99 1

<br>
测试结果如下： 

     train success with 301923 times !
      runing time: 7.15121s
      predict result:
      0.111 0.112 0
      0.001 0.999 1
      0.123 0.345 1
      0.123 0.456 1
      0.123 0.789 1
      0.234 0.567 1
      0.234 0.678 1
      0.387 0.387 0
      0.616 0.717 1
      0.555 0.555 0



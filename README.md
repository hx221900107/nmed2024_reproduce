# nmed2024_reproduce
论文中的数据集均需要进行申请，尝试提交了OASIS数据集的申请失败，可能是因为NIH的政策对中国等地区禁止访问。本复现使用论文中在huggingface上部署的demo，按照以下流程可以在windows和linux上完整运行。

下载download.py，运行：

  ```
  pip install huggingface_hub
  python download.py
  ```

然后会自动下载nmed2024文件，进入nmed2024文件夹：

  ```
  cd .\nmed2024\  
  ```

将app.py中第30行中的```ckpt_path = 'ckpt_swinunetr_stripped_MNI.pt'```替换为ckpt_swinunetr_stripped_MNI.pt文件的实际路径，然后运行：

  ```
  pip install -r requirements.txt
  pip install streamlit
  streamlit run app.py       
  ```
运行成功会弹出加载demo模型的网页界面，在其中可以通过网页填写数据对病例进行预测。


原论文github地址：
  ```
  https://github.com/vkola-lab/nmed2024.git
  ```


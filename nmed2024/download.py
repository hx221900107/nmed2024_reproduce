from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="vkola-lab/nmed2024",
    repo_type="space",  # 重要：指定为 space
    local_dir="./nmed2024",
)
print("下载完成！")

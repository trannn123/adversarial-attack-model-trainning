import random
import lief
import numpy as np
from Feature_space_attack import extractor, model

input_path = r"D:/malware_sample/sorel-dataset-PE-bin/0003840cfc486058c7e6de0d7e14d3683982e37a61e0d4a883409a22f980ef13.bin"
output_path = "modified.exe"

# load pe
binary = lief.parse(input_path)
if not binary:
    raise Exception("Cannot parse PE file")
print("Load PE thành công")

# load features
with open(input_path, "rb") as f:
    bytez = f.read()

raw = extractor.raw_features(bytez) # lấy thông tin PE thô
x_mal = extractor.process_raw_features(raw) # chuyển thành vector số
x_mal = np.pad(x_mal, (0, max(0, 2568 - len(x_mal)))) # đảm bảo 2568

x_adv = np.load("x_adv.npy") # file feature vector sau attack

diff = x_adv - x_mal # >0 thì feature tăng, <0 thì feature giảm
important_changes = np.argsort(np.abs(diff))[-20:] # lấy 20 feature thay đổi mạnh

print("Top feature:", important_changes)

# modify
#sửa tối đa 5 feature trong 1 file
MAX_MOD = 5
count = 0

for i in important_changes:
    if i <= 0 or i >= 2568:
        continue

    if count >= MAX_MOD:
        break

    change = diff[i]

    # feature tăng
    if change > 0:

        print(f"[+] Feature {i}")

        if len(binary.sections) > 0:
            # lấy section đầu tiên (.text hoặc .rdata)
            sec = binary.sections[0]

            content = list(sec.content)
            # tăng entropy, thay đổi pattern file
            for j in range(min(200, len(content))):
                content[j] = (content[j] + random.randint(0, 3)) % 256

            sec.content = content
            count += 1


    # feature giảm
    elif change < 0:

        print(f"[-] Feature {i}")

        if len(binary.sections) > 1:
            sec = binary.sections[1]

            content = list(sec.content)
            # giảm entropy, giảm thông tin trong section
            for j in range(min(200, len(content))):
                content[j] = 0x00

            sec.content = content
            count += 1

# tạo lại file pe kết quả
config = lief.PE.Builder.config_t()

config.imports = False
config.relocations = True
config.resources = True
config.tls = True
builder = lief.PE.Builder(binary, config)
builder.build()
builder.write(output_path)

print("Đã tạo modified.exe")

# kiểm tra
with open(output_path, "rb") as f:
    bytez = f.read()

raw = extractor.raw_features(bytez)
x_new = extractor.process_raw_features(raw)
x_new = np.pad(x_new, (0, max(0, 2568 - len(x_new))))

prob = model.predict([x_new])[0]

print("Final probability:", prob)
# from flask import Flask, render_template, request, redirect, url_for
# import os
# from inference import load_model_and_breed_info, predict

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB上传限制
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# # 确保上传文件夹存在
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # 全局加载模型和品种信息（只加载一次）
# try:
#     model, device, breed_info_map, label2id = load_model_and_breed_info()
#     print(f"模型加载成功，使用设备: {device}")
#     print(f"加载品种信息数量: {len(breed_info_map)}")
# except Exception as e:
#     print(f"模型加载失败: {e}")
#     model = None

# def allowed_file(filename):
#     """检查文件是否允许上传"""
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     """主页，显示上传表单"""
#     if model is None:
#         return render_template('index.html', error='模型加载失败，请检查best_model.pt文件是否存在')
    
#     if request.method == 'POST':
#         # 检查是否有文件上传
#         if 'file' not in request.files:
#             return render_template('index.html', error='未检测到文件，请选择图片后上传')
        
#         file = request.files['file']
        
#         # 如果用户没有选择文件
#         if file.filename == '':
#             return render_template('index.html', error='未选择文件，请选择图片后上传')
        
#         # 如果文件有效
#         if file and allowed_file(file.filename):
#             # 保存文件
#             filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(filename)
            
#             # 执行预测，传递品种信息映射表
#             results, image = predict(filename, model, device, breed_info_map, label2id)
            
#             # 如果预测失败
#             if results is None:
#                 os.remove(filename)  # 删除上传的文件
#                 return render_template('index.html', error='图片处理失败，请尝试使用其他图片')
            
#             # 返回结果页面，包含完整品种信息
#             return render_template('result.html', 
#                                  results=results, 
#                                  filename=file.filename)
        
#         else:
#             return render_template('index.html', error='不支持的文件类型，请上传PNG、JPG或GIF格式的图片')
    
#     # GET请求，显示上传表单
#     return render_template('index.html')

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     """提供上传图片的访问"""
#     return redirect(url_for('static', filename='uploads/' + filename), code=301)

# if __name__ == '__main__':
#     # 启动Web应用
#     app.run(debug=True)
import os
from flask import Flask, render_template, request
from inference import load_model_and_breed_info, predict, load_gpt_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB上传限制
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局加载模型和品种信息（只加载一次）
try:
    # 加载 ViT 模型和品种信息
    model, device, breed_info_map, label2id = load_model_and_breed_info()
    print(f"模型加载成功，使用设备: {device}")
    print(f"加载品种信息数量: {len(breed_info_map)}")
    
    # 加载 GPT 模型
    gpt_model, encode, decode = load_gpt_model("GPT/output/nanogpt_lan_final.pth", device=device)
    print(f"GPT 模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None
    gpt_model = None

def allowed_file(filename):
    """检查文件是否允许上传"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    """主页，显示上传表单"""
    if model is None or gpt_model is None:
        return render_template('index.html', error='模型加载失败，请检查模型文件是否存在')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='未检测到文件，请选择图片后上传')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='未选择文件，请选择图片后上传')
        
        if file and allowed_file(file.filename):
            # 保存文件到 static/uploads
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # 执行预测
            results = predict(filename, model, device, breed_info_map, label2id, gpt_model, encode, decode)
            
            if not results:
                os.remove(filename)
                return render_template('index.html', error='图片处理失败，请尝试使用其他图片')
            
            return render_template('result.html', results=results, filename=file.filename)  # Ensure results are passed
        
        else:
            return render_template('index.html', error='不支持的文件类型，请上传PNG、JPG或GIF格式的图片')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

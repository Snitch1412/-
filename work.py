import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.dialogs import Messagebox
from ttkbootstrap.scrolled import ScrolledText
from csv import *
import numpy as np
import sympy as sp
import os
import sys
import re
import warnings
import threading
import time
from functools import wraps

# ======================================
# 全局变量定义
# ======================================
ocr_reader = None
ocr_loading = False
ocr_initialized = False
Amatrix = None
Bmatrix = None
Bnum = None
C_matrix = None
current_eigenvals = None
current_eigenvects = None
current_scalar_result = None
calculation_history = []
fig = None
filedialog = None 
plt = None 
FigureCanvasTkAgg = None  
arrow = None  
shutdown_event = threading.Event()
# ======================================
# 懒加载装饰器
# ======================================
def lazy_load_ocr(func):
    """懒加载装饰器，确保OCR模块在使用前被加载"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        global ocr_reader, ocr_initialized
        
        # 如果OCR已经初始化，直接执行函数
        if ocr_initialized and ocr_reader is not None:
            return func(*args, **kwargs)
        
        # 如果正在加载中，提示用户等待
        if ocr_loading:
            raise Exception("OCR模块正在初始化中，请稍后再试...")
        
        # 如果还未初始化，抛出异常提示用户先初始化
        raise Exception("OCR模块未初始化，请先初始化OCR功能")
    
    return wrapper

# ======================================
# OCR 相关功能
# ======================================
class OCRManager:
    """OCR管理器，实现懒加载和状态管理"""
    
    def __init__(self):
        self.reader = None
        self.loading = False
        self.initialized = False
        self.load_callbacks = []
    
    def is_available(self):
        """检查OCR是否可用"""
        return self.initialized and self.reader is not None
    
    def is_loading(self):
        """检查是否正在加载"""
        return self.loading
    
    def add_load_callback(self, callback):
        """添加加载完成回调"""
        if callback not in self.load_callbacks:
            self.load_callbacks.append(callback)
    
    def remove_load_callback(self, callback):
        """移除加载回调"""
        if callback in self.load_callbacks:
            self.load_callbacks.remove(callback)
    
    def _notify_callbacks(self, success, error_msg=None):
        """通知所有回调函数"""
        for callback in self.load_callbacks.copy():
            try:
                callback(success, error_msg)
            except Exception as e:
                print(f"回调函数执行失败: {e}")
    
    def lazy_initialize(self, parent, callback=None, show_progress=True):
        """懒加载初始化OCR"""
        # 如果已经初始化，直接回调
        if self.is_available():
            if callback:
                callback(True, None)
            return True
        
        # 如果正在加载，只添加回调
        if self.is_loading():
            if callback:
                self.add_load_callback(callback)
            return False
        
        # 开始加载
        if callback:
            self.add_load_callback(callback)
        
        if show_progress:
            self._load_with_progress(parent)
        else:
            self._load_silent()
        
        return False
    
    def _load_with_progress(self, parent):
        """带进度条的加载"""
        self.loading = True
        
        # 创建进度对话框
        progress_window, progress_bar, status_label = self._create_progress_dialog(parent)
        
        def update_status(text):
            """安全地更新状态"""
            def _update():
                if progress_window.winfo_exists():
                    status_label.config(text=text)
            parent.after(0, _update)
        
        def close_progress_and_notify(success, error_msg=None):
            """关闭进度窗口并通知回调"""
            def _close():
                try:
                    if progress_window.winfo_exists():
                        progress_bar.stop()
                        progress_window.destroy()
                except tk.TclError:
                    pass
                finally:
                    self.loading = False
                    self._notify_callbacks(success, error_msg)
                    self.load_callbacks.clear()
            parent.after(0, _close)
        
        def load_thread():
            try:
                update_status("检查EasyOCR依赖...")
                time.sleep(0.1)
                
                # 动态导入EasyOCR
                update_status("导入EasyOCR库...")
                import easyocr
                
                update_status("初始化OCR识别器...")
                
                # 尝试GPU模式，失败则使用CPU
                try:
                    update_status("尝试GPU模式...")
                    self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
                    update_status("GPU模式初始化成功!")
                except Exception as gpu_error:
                    update_status("GPU模式失败，切换到CPU模式...")
                    self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                    update_status("CPU模式初始化成功!")
                
                time.sleep(0.5)
                self.initialized = True
                
                # 更新全局变量（保持向后兼容）
                global ocr_reader, ocr_initialized
                ocr_reader = self.reader
                ocr_initialized = True
                
                close_progress_and_notify(True)
                
            except Exception as e:
                close_progress_and_notify(False, f"OCR初始化失败: {str(e)}")
        
        # 启动加载线程
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
    
    def _load_silent(self):
        """静默加载（后台加载）"""
        self.loading = True
        
        def load_thread():
            try:
                import easyocr
                
                # 尝试GPU，失败则CPU
                try:
                    self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
                except:
                    self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
                
                self.initialized = True
                
                # 更新全局变量
                global ocr_reader, ocr_initialized
                ocr_reader = self.reader
                ocr_initialized = True
                
                self.loading = False
                self._notify_callbacks(True)
                self.load_callbacks.clear()
                
            except Exception as e:
                self.loading = False
                self._notify_callbacks(False, str(e))
                self.load_callbacks.clear()
        
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
    
    def _create_progress_dialog(self, parent):
        """创建进度对话框"""
        progress_window = ttk.Toplevel(parent)
        progress_window.title("初始化OCR")
        progress_window.geometry("450x180")
        progress_window.resizable(False, False)
        
        # 设置窗口居中
        progress_window.transient(parent)
        progress_window.grab_set()
        
        # 创建界面元素
        main_frame = ttk.Frame(progress_window, padding=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        ttk.Label(main_frame, text="正在初始化OCR模块", 
                  font=("Arial", 14, "bold")).pack(pady=10)
        
        progress_bar = ttk.Progressbar(main_frame, mode='indeterminate', 
                                       bootstyle="info", length=350)
        progress_bar.pack(pady=10)
        progress_bar.start(10)
        
        status_label = ttk.Label(main_frame, text="准备加载EasyOCR...", 
                                font=("Arial", 10))
        status_label.pack(pady=5)
        
        ttk.Label(main_frame, text="首次使用需要下载模型文件，请耐心等待", 
                  font=("Arial", 9), foreground="gray").pack(pady=5)
        
        return progress_window, progress_bar, status_label
    
    @lazy_load_ocr
    def recognize_text(self, image_path):
        """识别图像中的文字"""
        return self._perform_ocr(image_path)
    
    def _perform_ocr(self, image_path):
        """执行OCR识别"""
        if not self.reader:
            raise Exception("OCR识别器未初始化")
        
        try:
            # 执行OCR识别
            results = self.reader.readtext(image_path)
            
            # 处理识别结果，提取数字矩阵
            matrix_data = []
            
            # 按行分组识别结果（根据y坐标）
            lines = {}
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 只保留置信度较高的结果
                    # 计算边界框的中心y坐标
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    
                    # 将相近的y坐标归为同一行
                    line_key = round(y_center / 10) * 10
                    
                    if line_key not in lines:
                        lines[line_key] = []
                    lines[line_key].append((bbox[0][0], text))
            
            # 按行处理数据
            for line_y in sorted(lines.keys()):
                # 按x坐标排序同一行的文本
                line_texts = sorted(lines[line_y], key=lambda x: x[0])
                
                row_data = []
                for _, text in line_texts:
                    # 从文本中提取数字（支持正负数、小数）
                    numbers = re.findall(r'-?\d+\.?\d*', text)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            row_data.append(num)
                        except ValueError:
                            continue
                
                if row_data:
                    matrix_data.append(row_data)
            
            # 如果没有识别到矩阵数据，尝试简单的文本分割方法
            if not matrix_data:
                all_text = ' '.join([text for (_, text, conf) in results if conf > 0.5])
                lines = all_text.split('\n')
                for line in lines:
                    numbers = re.findall(r'-?\d+\.?\d*', line)
                    if numbers:
                        row_data = [float(num) for num in numbers]
                        matrix_data.append(row_data)
            
            return matrix_data
            
        except Exception as e:
            raise Exception(f"图像识别失败: {str(e)}")

# 创建全局OCR管理器实例
ocr_manager = OCRManager()

# ======================================
# 向后兼容的函数接口
# ======================================
def load_ocr_with_progress(parent, callback):
    """向后兼容的OCR加载函数"""
    global ocr_loading
    ocr_loading = ocr_manager.is_loading()
    
    def wrapped_callback(success, error_msg):
        global ocr_loading
        ocr_loading = False
        callback(success, error_msg)
    
    return ocr_manager.lazy_initialize(parent, wrapped_callback, show_progress=True)

@lazy_load_ocr
def recognize_image_text(image_path):
    """向后兼容的图像识别函数"""
    return ocr_manager._perform_ocr(image_path)

def safe_initialize_ocr(parent, callback=None):
    """安全的OCR初始化，带用户友好提示"""
    def on_ocr_loaded(success, error_msg):
        if success:
            Messagebox.show_info("OCR初始化成功!", 
                               "OCR模块已成功加载，可以开始使用图像识别功能。")
        else:
            Messagebox.show_error("OCR初始化失败", 
                                f"{error_msg}\n\n提示：\n"
                                "1. 确保已安装 easyocr 库: pip install easyocr\n"
                                "2. 首次使用需要下载模型文件，请确保网络连接正常\n"
                                "3. 如果GPU初始化失败，会自动使用CPU模式\n"
                                "4. 模型文件较大，首次下载可能需要几分钟时间")
        
        if callback:
            callback(success, error_msg)
    
    return ocr_manager.lazy_initialize(parent, on_ocr_loaded, show_progress=True)

# ======================================
# 便捷函数
# ======================================
def is_ocr_available():
    """检查OCR是否可用"""
    return ocr_manager.is_available()

def is_ocr_loading():
    """检查OCR是否正在加载"""
    return ocr_manager.is_loading()

def preload_ocr_silent(callback=None):
    """后台静默预加载OCR（不显示进度条）"""
    return ocr_manager.lazy_initialize(None, callback, show_progress=False)

def get_ocr_status():
    """获取OCR状态信息"""
    if ocr_manager.is_available():
        return "已初始化"
    elif ocr_manager.is_loading():
        return "初始化中..."
    else:
        return "未初始化"

# ======================================
# 使用示例
# ======================================
def example_usage(parent):
    """使用示例"""
    
    # 方式1: 显式初始化（推荐用于用户主动操作）
    def on_init_complete(success, error_msg):
        if success:
            print("OCR初始化成功，可以开始识别图像")
            # 现在可以安全使用 recognize_image_text()
        else:
            print(f"初始化失败: {error_msg}")
    
    safe_initialize_ocr(parent, on_init_complete)
    
    # 方式2: 懒加载（在需要时自动初始化）
    try:
        # 这会触发懒加载，如果OCR未初始化会抛出异常
        result = recognize_image_text("path/to/image.jpg")
        print("识别结果:", result)
    except Exception as e:
        print(f"识别失败: {e}")
        # 提示用户先初始化OCR
        safe_initialize_ocr(parent)
    
    # 方式3: 检查状态后使用
    if is_ocr_available():
        result = recognize_image_text("path/to/image.jpg")
    elif is_ocr_loading():
        print("OCR正在初始化中，请稍后...")
    else:
        print("OCR未初始化，开始初始化...")
        safe_initialize_ocr(parent)
    
    # 方式4: 后台预加载（适合应用启动时）
    preload_ocr_silent(lambda success, error: print(f"后台加载: {'成功' if success else '失败'}"))

# ======================================
# 矩阵处理辅助函数
# ======================================
def get_file_type_by_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()

def format_sympy_output(expr):
    """格式化sympy表达式输出，支持分数和复数"""
    if hasattr(expr, 'evalf'):
        # 如果是有理数，保持分数形式
        if isinstance(expr, sp.Rational):
            return str(expr)
        # 如果是复数
        elif expr.has(sp.I):
            return str(expr)
        # 其他情况显示小数
        else:
            return f"{float(expr.evalf()):8.3f}"
    else:
        return f"{expr:8.3f}"

def matrix_to_numpy(sympy_matrix):
    """将sympy矩阵转换为numpy数组，处理复数"""
    if sympy_matrix is None:
        return None
    
    try:
        # 转换为浮点数数组
        np_matrix = np.zeros((sympy_matrix.rows, sympy_matrix.cols), dtype=complex)
        for i in range(sympy_matrix.rows):
            for j in range(sympy_matrix.cols):
                element = sympy_matrix[i, j]
                if element.has(sp.I):
                    # 处理复数
                    np_matrix[i, j] = complex(element.evalf())
                else:
                    # 处理实数
                    np_matrix[i, j] = float(element.evalf())
        
        # 如果所有元素都是实数，转换为实数数组
        if np.all(np.imag(np_matrix) == 0):
            np_matrix = np.real(np_matrix)
            
        return np_matrix
    except Exception as e:
        print(f"转换矩阵时出错: {e}")
        return None

# ======================================
# 矩阵导入功能
# ======================================
def process_image_matrix(file_path, entry_widget, text_widget, matrix_name):
    """处理图片矩阵的通用函数"""
    global Amatrix, Bmatrix
    
    def on_ocr_loaded(success, error_msg):
        if not success:
            Messagebox.show_error("错误", error_msg)
            return
            
        try:
            # 使用EasyOCR识别图片中的矩阵
            matrix_data = recognize_image_text(file_path)
            
            if not matrix_data:
                Messagebox.show_warning("警告", "未能从图片中识别出矩阵数据，请检查图片质量")
                return
                
            # 确保所有行的长度相同（补齐或截断）
            max_len = max(len(row) for row in matrix_data)
            normalized_data = []
            for row in matrix_data:
                if len(row) < max_len:
                    row.extend([0.0] * (max_len - len(row)))  # 用0补齐
                elif len(row) > max_len:
                    row = row[:max_len]  # 截断
                normalized_data.append(row)
                
            matrix = np.array(normalized_data)
            
            # 更新全局变量
            if matrix_name == 'A':
                Amatrix = matrix
            else:
                Bmatrix = matrix
            
            # 显示矩阵
            text_widget.delete(1.0, END)
            for row in matrix:
                for element in row:
                    text_widget.insert(END, f"{round(element, 1):8}")
                text_widget.insert(END, "\n")
                
            Messagebox.show_info("成功", f"矩阵{matrix_name}已导入，形状: {matrix.shape}")
                
        except Exception as e:
            Messagebox.show_error("错误", f"图片识别失败: {str(e)}")
    
    # 检查是否需要初始化OCR
    load_ocr_with_progress(root, on_ocr_loaded)

def A():
    lazy_imports()
    global Amatrix
    file_pathA = filedialog.askopenfilename(title="选择文件", 
                                          filetypes=[("所有文件", "*.*"), 
                                                   ("文本文件", "*.txt"), 
                                                   ("CSV 文件", "*.csv"), 
                                                   ("图片文件", "*.jpg;*.png;*.jpeg;*.bmp")])
    if not file_pathA:
        return
        
    EntryA.delete(0, END)
    EntryA.insert(0, file_pathA)
    typeA = get_file_type_by_extension(file_pathA)
    
    try:
        if typeA == '.txt' or typeA == '.csv':
            # 先尝试直接读取
            try:
                matrixA = np.loadtxt(file_pathA, ndmin=2)
            except:
                # 如果失败，尝试检测分隔符
                with open(file_pathA, 'r') as f:
                    sample = f.read(1024)
                    dialect = Sniffer().sniff(sample)
                    delimiter = dialect.delimiter
                matrixA = np.loadtxt(file_pathA, delimiter=delimiter, ndmin=2)
                
            # 更新全局变量
            Amatrix = matrixA
            
            # 显示矩阵
            textA.delete(1.0, END)
            for row in matrixA:
                for element in row:
                    textA.insert(END, f"{round(element, 1):8}")
                textA.insert(END, "\n")
                
            Messagebox.show_info("成功", f"矩阵A已导入，形状: {matrixA.shape}")
                
        elif typeA in ['.jpg', '.png', '.jpeg', '.bmp']:
            # 使用新的图片处理函数
            process_image_matrix(file_pathA, EntryA, textA, 'A')
            
    except Exception as e:
        Messagebox.show_error("错误", f"读取文件失败: {str(e)}")

def B():
    global Bmatrix
    file_pathB = filedialog.askopenfilename(title="选择文件",
                                          filetypes=[("所有文件", "*.*"), 
                                                   ("文本文件", "*.txt"), 
                                                   ("CSV 文件", "*.csv"), 
                                                   ("图片文件", "*.jpg;*.png;*.jpeg;*.bmp")])
    if not file_pathB:
        return
        
    EntryB.delete(0, END)
    EntryB.insert(0, file_pathB)
    typeB = get_file_type_by_extension(file_pathB)
    
    try:
        if typeB == '.txt' or typeB == '.csv':
            try:
                matrixB = np.loadtxt(file_pathB, ndmin=2)
            except:
                with open(file_pathB, 'r') as f:
                    sample = f.read(1024)
                    dialect = Sniffer().sniff(sample)
                    delimiter = dialect.delimiter
                matrixB = np.loadtxt(file_pathB, delimiter=delimiter, ndmin=2)
                
            # 更新全局变量
            Bmatrix = matrixB
            
            # 显示矩阵
            textB.delete(1.0, END)
            for row in matrixB:
                for element in row:
                    textB.insert(END, f"{round(element, 1):8}")
                textB.insert(END, "\n")
                
            Messagebox.show_info("成功", f"矩阵B已导入，形状: {matrixB.shape}")
                
        elif typeB in ['.jpg', '.png', '.jpeg', '.bmp']:
            # 使用新的图片处理函数
            process_image_matrix(file_pathB, EntryB, textB, 'B')
            
    except Exception as e:
        Messagebox.show_error("错误", f"读取文件失败: {str(e)}")

# ======================================
# 矩阵生成功能
# ======================================
def creat():
    try:
        # 精准验证矩阵大小输入
        try:
            rows = int(spinboxA.get())
            cols = int(spinboxB.get())
        except ValueError:
            Messagebox.show_error("矩阵大小必须为整数！请检查行数和列数输入。")
            return
        
        # 验证矩阵大小是否合理
        if rows <= 0 or cols <= 0:
            Messagebox.show_error("矩阵大小必须为正整数！")
            return
        
        size = (rows, cols)
        
        # 验证分布参数输入
        try:
            if comboB.get() == "正态分布":
                mean = float(entryA.get())
                std = float(entryB.get())
                if std <= 0:
                    Messagebox.show_error("正态分布的标准差必须大于0！")
                    return
                matrix = np.random.normal(mean, std, size)
                
            elif comboB.get() == "均匀分布":
                low = float(entryA.get())
                high = float(entryB.get())
                if low >= high:
                    Messagebox.show_error("均匀分布的上界必须大于下界！")
                    return
                matrix = np.random.uniform(low, high, size)
                
            elif comboB.get() == "泊松分布":
                lam = float(entryA.get())
                if lam <= 0:
                    Messagebox.show_error("泊松分布的参数λ必须大于0！")
                    return
                matrix = np.random.poisson(lam, size)
                
            elif comboB.get() == "指数分布":
                scale = float(entryA.get())
                if scale <= 0:
                    Messagebox.show_error("指数分布的尺度参数必须大于0！")
                    return
                matrix = np.random.exponential(scale, size)
            else:
                Messagebox.show_error("请选择有效的分布类型！")
                return
                
        except ValueError:
            Messagebox.show_error("分布参数必须为有效数字！请检查参数输入。")
            return
        
        # 显示生成的矩阵
        target_text = textA if matrix_value_dist[matrix_value.get()] == "矩阵A" else textB
        target_text.delete(1.0, END)
        
        for row in matrix:
            for element in row:
                target_text.insert(END, f"{round(element, 2):8}")
            target_text.insert(END, "\n")
            
    except Exception as e:
        Messagebox.show_error(f"生成矩阵时发生未知错误: {str(e)}")


# ======================================
# 矩阵操作和计算
# ======================================
def ensureA():
    global Amatrix
    try:
        text_contentA = textA.get("1.0", END).strip()
        if not text_contentA:
            Messagebox.show_error("矩阵A为空")
            return
            
        rows = text_contentA.split("\n")
        amatrix = []
        
        for row in rows:
            if row.strip():  # 跳过空行
                row_elements = []
                for num_str in row.split():
                    try:
                        # 支持分数、复数和普通数字
                        if '/' in num_str and 'I' not in num_str:  # 分数
                            element = sp.Rational(num_str)
                        elif 'I' in num_str or 'i' in num_str:  # 复数
                            element =  sp.sympify(num_str.replace('i', 'I'))
                        else:  # 普通数字
                            element =  sp.sympify(num_str)
                        row_elements.append(element)
                    except (sp.SympifyError, ValueError):
                        Messagebox.show_error(f"无法解析数字: {num_str}")
                        return
                if row_elements:
                    amatrix.append(row_elements)
        
        if not amatrix:
            Messagebox.show_error("矩阵A为空")
            return
            
        Amatrix = sp.Matrix(amatrix)

        
    except Exception as e:
        Messagebox.show_error(f"确定矩阵A失败: {str(e)}")

def ensureB():
    global Bnum, Bmatrix
    try:
        text_contentB = textB.get("1.0", END).strip()
        if not text_contentB:
            Messagebox.show_error("错误, 矩阵B/数b为空")
            return
        
        # 检测是否为单个数字（没有换行符且没有多个空格分隔的数字）
        is_single_number = '\n' not in text_contentB and len(text_contentB.split()) == 1
        
        if is_single_number:
            # 如果是单个数字，保存为Bnum
            try:
                if '/' in text_contentB and 'I' not in text_contentB:  # 分数
                    Bnum = sp.Rational(text_contentB.strip())
                elif 'I' in text_contentB or 'i' in text_contentB:  # 复数
                    Bnum =  sp.sympify(text_contentB.strip().replace('i', 'I'))
                else:  # 普通数字
                    Bnum =  sp.sympify(text_contentB.strip())
            except (sp.SympifyError, ValueError):
                Messagebox.show_error(f"无法解析数字: {text_contentB}")
                return
        else:
            # 如果不是单个数字，Bnum设为None
            Bnum = None
        
        # 无论如何都解析为矩阵Bmatrix
        rows = text_contentB.split("\n")
        bmatrix = []
        
        for row in rows:
            if row.strip():
                row_elements = []
                for num_str in row.split():
                    try:
                        if '/' in num_str and 'I' not in num_str:
                            element = sp.Rational(num_str)
                        elif 'I' in num_str or 'i' in num_str:
                            element =  sp.sympify(num_str.replace('i', 'I'))
                        else:
                            element =  sp.sympify(num_str)
                        row_elements.append(element)
                    except (sp.SympifyError, ValueError):
                        Messagebox.show_error(f"无法解析数字: {num_str}")
                        return
                if row_elements:
                    bmatrix.append(row_elements)
        
        if bmatrix:
            Bmatrix = sp.Matrix(bmatrix)
            
        else:
            Messagebox.show_error("无法解析输入内容")
            
    except Exception as e:
        Messagebox.show_error(f"确定矩阵B/数失败: {str(e)}")

def operation():
    ensureA()
    global Amatrix, Bmatrix, Bnum, C_matrix, current_eigenvals, current_eigenvects,current_scalar_result
    C_matrix = None
    current_eigenvals = None
    current_eigenvects = None
    current_scalar_result = None
    
    textC.config(state="normal")
    textC.delete(1.0, END)
    if Amatrix is None:
        return
    try:
        if comboC.get() == "加法A+B":
            ensureB()
            if Bmatrix is None:
                return
            if Amatrix.shape != Bmatrix.shape:
                Messagebox.show_error(f"矩阵维度不匹配：A({Amatrix.rows}×{Amatrix.cols}) 与 B({Bmatrix.rows}×{Bmatrix.cols})")
                return
            Cmatrix = Amatrix + Bmatrix
            
        elif comboC.get() == "减法A-B":
            ensureB()
            if Bmatrix is None:
                return
            if Amatrix.shape != Bmatrix.shape:
                Messagebox.show_error(f"矩阵维度不匹配：A({Amatrix.rows}×{Amatrix.cols}) 与 B({Bmatrix.rows}×{Bmatrix.cols})")
                return
            Cmatrix = Amatrix - Bmatrix
            
        elif comboC.get() == "数乘bxA":
            ensureB()
            if Bmatrix is None:
                return
            if Bnum is None:
                Messagebox.show_error("请先确定数值b")
                return
            Cmatrix = Bnum * Amatrix
            
        elif comboC.get() == "乘法AxB":
            ensureB()
            if Bmatrix is None:
                return
            if Amatrix.cols != Bmatrix.rows:
                Messagebox.show_error(f"矩阵乘法维度不匹配：A的列数({Amatrix.cols}) ≠ B的行数({Bmatrix.rows})")
                return
            Cmatrix = Amatrix * Bmatrix
            
        elif comboC.get() == "A的秩":
            try:
                result = Amatrix.rank()
                current_scalar_result = result  # 新增：存储标量结果
                textC.insert("1.0", f"矩阵A的秩: {result}")
            except Exception as e:
                Messagebox.show_error(f"计算矩阵秩失败: {str(e)}")
                return
            
        elif comboC.get() == "A的转置":
            Cmatrix = Amatrix.T
            
        elif comboC.get() == "A的行列式":
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"只有方阵才能计算行列式，当前矩阵为{Amatrix.rows}×{Amatrix.cols}")
                return
            try:
                result = Amatrix.det()
                # 添加这行代码：将结果存储到全局变量 ↓↓↓
                current_scalar_result = result  # 新增此行
                textC.insert("1.0", f"矩阵A的行列式: {format_sympy_output(result)}")
            except Exception as e:
                Messagebox.show_error(f"计算行列式失败: {str(e)}")
                return
            
        elif comboC.get() == "A的逆矩阵":
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"只有方阵才能计算逆矩阵，当前矩阵为{Amatrix.rows}×{Amatrix.cols}")
                return
            try:
                det_A = Amatrix.det()
                if det_A == 0:
                    Messagebox.show_error("矩阵不可逆（行列式为0）")
                    return
                Cmatrix = Amatrix.inv()
            except Exception as e:
                Messagebox.show_error(f"计算逆矩阵失败: {str(e)}")
                return
        elif comboC.get() == "A的伴随矩阵":
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"只有方阵才有伴随矩阵，当前矩阵为{Amatrix.rows}×{Amatrix.cols}")
                return
            try:
                Cmatrix = Amatrix.adjugate()
            except Exception as e:
                Messagebox.show_error(f"计算伴随矩阵失败: {str(e)}")
                return
             
        elif comboC.get() == "幂A^b":
            ensureB()
            if Bnum is None:
                Messagebox.show_error("请先确定幂次数b")
                return
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"只有方阵才能计算幂，当前矩阵为{Amatrix.rows}×{Amatrix.cols}")
                return
            try:
                power = int(Bnum)
                if power < 0:
                    # 负幂需要先检查矩阵是否可逆
                    det_A = Amatrix.det()
                    if det_A == 0:
                        Messagebox.show_error("矩阵不可逆，无法计算负幂")
                        return
                Cmatrix = Amatrix ** power
            except ValueError:
                Messagebox.show_error("幂次数必须是整数")
                return
            except Exception as e:
                Messagebox.show_error(f"计算矩阵幂失败: {str(e)}")
                return
            
        elif comboC.get() == "A的特征值":
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"只有方阵才能计算特征值，当前矩阵为{Amatrix.rows}×{Amatrix.cols}")
                return
            try:
                current_eigenvals = Amatrix.eigenvals()
                if not current_eigenvals:
                    textC.insert(END, "未找到特征值")
                else:
                    for val, mult in current_eigenvals.items():
                        textC.insert(END, f"特征值: {format_sympy_output(val)}, 重数: {mult}\n")
            except Exception as e:
                Messagebox.show_error(f"计算特征值失败: {str(e)}")
                return
            
        elif comboC.get() == "A的特征向量":
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"只有方阵才能计算特征向量，当前矩阵为{Amatrix.rows}×{Amatrix.cols}")
                return
            try:
                current_eigenvects = Amatrix.eigenvects()
                if not current_eigenvects:
                    textC.insert(END, "未找到特征向量")
                else:
                    for val, mult, vects in current_eigenvects:
                        textC.insert(END, f"特征值: {format_sympy_output(val)}, 重数: {mult}\n")
                        for i, vect in enumerate(vects):
                            textC.insert(END, f"特征向量{i+1}: ")
                            for element in vect:
                                textC.insert(END, f"{format_sympy_output(element)} ")
                            textC.insert(END, "\n")
                        textC.insert(END, "\n")
            except Exception as e:
                Messagebox.show_error(f"计算特征向量失败: {str(e)}")
                return
        
        elif comboC.get() == "求解AX=B":
            ensureB()
            if Bmatrix is None:
                return
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"系数矩阵A必须是方阵，当前为{Amatrix.rows}×{Amatrix.cols}")
                return
            if Amatrix.rows != Bmatrix.rows:
                Messagebox.show_error(f"维度不匹配：A的行数({Amatrix.rows}) ≠ B的行数({Bmatrix.rows})")
                return
            try:
                det_A = Amatrix.det()
                if det_A == 0:
                    Messagebox.show_error("矩阵A不可逆（行列式为0），方程组无唯一解")
                    return
                Cmatrix = Amatrix.inv() * Bmatrix
            except Exception as e:
                Messagebox.show_error(f"求解线性方程组失败: {str(e)}")
                return
        
        elif comboC.get() == "交换A的1和b行":
            ensureB()
            if Bnum is None:
                Messagebox.show_error("请先确定要交换的行号b")
                return
            try:
                b_row = int(Bnum)
                if b_row < 1 or b_row > Amatrix.rows:
                    Messagebox.show_error(f"行号b必须在1到{Amatrix.rows}之间，当前输入: {b_row}")
                    return
                if b_row == 1:
                    Messagebox.show_warning("交换第1行和第1行，矩阵不变")
                Cmatrix = Amatrix.copy()
                # 交换第1行（索引0）和第b行（索引b-1）
                for j in range(Cmatrix.cols):
                    Cmatrix[0, j], Cmatrix[b_row-1, j] = Cmatrix[b_row-1, j], Cmatrix[0, j]
            except ValueError:
                Messagebox.show_error(f"行号b必须是整数，当前输入: {Bnum}")
                return
            except Exception as e:
                Messagebox.show_error(f"交换行失败: {str(e)}")
                return
        
        elif comboC.get() == "交换A的1和b列":
            ensureB()
            if Bnum is None:
                Messagebox.show_error("请先确定要交换的列号b")
                return
            try:
                b_col = int(Bnum)
                if b_col < 1 or b_col > Amatrix.cols:
                    Messagebox.show_error(f"列号b必须在1到{Amatrix.cols}之间，当前输入: {b_col}")
                    return
                if b_col == 1:
                    Messagebox.show_warning("交换第1列和第1列，矩阵不变")
                Cmatrix = Amatrix.copy()
                # 交换第1列（索引0）和第b列（索引b-1）
                for i in range(Cmatrix.rows):
                    Cmatrix[i, 0], Cmatrix[i, b_col-1] = Cmatrix[i, b_col-1], Cmatrix[i, 0]
            except ValueError:
                Messagebox.show_error(f"列号b必须是整数，当前输入: {Bnum}")
                return
            except Exception as e:
                Messagebox.show_error(f"交换列失败: {str(e)}")
                return
        
        elif comboC.get() == "A的范数":
            try:
                # 计算Frobenius范数
                norm_result = sp.sqrt(sum([element**2 for element in Amatrix]))
                simplified_norm = sp.simplify(norm_result)
                textC.insert("1.0", f"Frobenius范数: {format_sympy_output(simplified_norm)}")
                current_scalar_result = simplified_norm
            except Exception as e:
                Messagebox.show_error(f"计算范数失败: {str(e)}")
                return
        
        elif comboC.get() == "A的迹":
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error(f"只有方阵才能计算迹，当前矩阵为{Amatrix.rows}×{Amatrix.cols}")
                return
            try:
                trace_result = Amatrix.trace()
                simplified_trace = sp.simplify(trace_result)
                textC.insert("1.0", f"矩阵的迹: {format_sympy_output(simplified_trace)}")
                current_scalar_result = simplified_trace
            except Exception as e:
                Messagebox.show_error(f"计算迹失败: {str(e)}")
                return
        
        elif comboC.get() == "A的共轭转置":
            try:
                # 使用sympy的共轭转置
                Cmatrix = Amatrix.H  # H表示共轭转置（Hermitian转置）
            except Exception as e:
                Messagebox.show_error(f"计算共轭转置失败: {str(e)}")
                return
        
        elif comboC.get() == "水平拼接":
            ensureB()
            if Amatrix.rows != Bmatrix.rows:
                Messagebox.show_error(f"水平拼接要求两矩阵行数相同：A({Amatrix.rows}行) ≠ B({Bmatrix.rows}行)")
                return
            try:
                Cmatrix = Amatrix.row_join(Bmatrix)
            except Exception as e:
                Messagebox.show_error(f"水平拼接失败: {str(e)}")
                return
        
        elif comboC.get() == "垂直拼接":
            ensureB()
            if Amatrix.cols != Bmatrix.cols:
                Messagebox.show_error(f"垂直拼接要求两矩阵列数相同：A({Amatrix.cols}列) ≠ B({Bmatrix.cols}列)")
                return
            try:
                Cmatrix = Amatrix.col_join(Bmatrix)
            except Exception as e:
                Messagebox.show_error(f"垂直拼接失败: {str(e)}")
                return
        
        elif comboC.get() == "A的LU分解":
            if Amatrix.rows != Amatrix.cols:
                Messagebox.show_error("LU分解仅适用于方阵")
                return
            try:
                L, U, _ = Amatrix.LUdecomposition()
                C_matrix = (L, U)
                textC.insert(END, "下三角矩阵 L:\n")
                for i in range(L.rows):
                    for j in range(L.cols):
                        textC.insert(END, format_sympy_output(L[i, j]) + " ")
                    textC.insert(END, "\n")
                textC.insert(END, "\n上三角矩阵 U:\n")
                for i in range(U.rows):
                    for j in range(U.cols):
                        textC.insert(END, format_sympy_output(U[i, j]) + " ")
                    textC.insert(END, "\n")
            except Exception as e:
                Messagebox.show_error(f"LU分解失败: {str(e)}")
                return
        
        elif comboC.get() == "A的QR分解":
            try:
                Q, R = Amatrix.QRdecomposition()
                C_matrix = (Q, R)
                textC.insert(END, "正交矩阵 Q:\n")
                for i in range(Q.rows):
                    for j in range(Q.cols):
                        textC.insert(END, format_sympy_output(Q[i, j]) + " ")
                    textC.insert(END, "\n")
                textC.insert(END, "\n上三角矩阵 R:\n")
                for i in range(R.rows):
                    for j in range(R.cols):
                        textC.insert(END, format_sympy_output(R[i, j]) + " ")
                    textC.insert(END, "\n")
            except Exception as e:
                Messagebox.show_error(f"QR分解失败: {str(e)}")
                return
        
        elif comboC.get() == "A的奇异值分解":
            try:
                np_A = matrix_to_numpy(Amatrix)
                if np_A is None:
                    Messagebox.show_error("无法将矩阵转换为数值进行SVD")
                    return
                U, S, Vh = np.linalg.svd(np_A)
                # 将S转为对角矩阵
                S_mat = np.zeros((U.shape[0], Vh.shape[0]))
                np.fill_diagonal(S_mat, S)
                
                # 转为sympy矩阵再赋值
                C_matrix = (sp.Matrix(U), sp.Matrix(S_mat), sp.Matrix(Vh))

                textC.insert(END, "U矩阵:\n")
                textC.insert(END, np.array2string(U, precision=3, suppress_small=True) + "\n\n")
                textC.insert(END, "奇异值 Σ:\n")
                textC.insert(END, str(S) + "\n\n")
                textC.insert(END, "V^H矩阵:\n")
                textC.insert(END, np.array2string(Vh, precision=3, suppress_small=True))
            except Exception as e:
                Messagebox.show_error(f"SVD分解失败: {str(e)}")
                return

        else:
            Messagebox.show_error(f"未知的操作类型: {comboC.get()}")
            return
        
        # 显示矩阵结果
        if 'Cmatrix' in locals() and Cmatrix is not None:
            try:
                for i in range(Cmatrix.rows):
                    for j in range(Cmatrix.cols):
                        element = Cmatrix[i, j]
                        textC.insert(END, format_sympy_output(element) + " ")
                    textC.insert(END, "\n")
                C_matrix = Cmatrix
            except Exception as e:
                Messagebox.show_error(f"显示结果失败: {str(e)}")
                return
                
    except Exception as e:
        Messagebox.show_error(f"计算过程中发生未知错误: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
   
    save_to_history()
    
    textC.config(state="disabled")

# ======================================
# 导出功能
# ======================================
def export_matrix_data(matrix, title):
    """导出矩阵数据为txt文件"""
    if matrix is None:
        return
    
    file_path = filedialog.asksaveasfilename(
        title=f"导出{title}",
        defaultextension=".txt",
        filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv")]
    )
    
    if not file_path:
        return
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}数据导出\n")
            f.write(f"矩阵大小: {matrix.rows}×{matrix.cols}\n")
            f.write("="*50 + "\n\n")
            
            # 写入矩阵数据
            f.write("矩阵数据:\n")
            for i in range(matrix.rows):
                for j in range(matrix.cols):
                    element = matrix[i, j]
                    f.write(f"{element}\t")
                f.write("\n")
            
            # 如果是CSV文件，添加CSV格式
            if file_path.endswith('.csv'):
                f.write("\n\nCSV格式:\n")
                for i in range(matrix.rows):
                    row_data = []
                    for j in range(matrix.cols):
                        element = matrix[i, j]
                        if element.has(sp.I):
                            row_data.append(str(element))
                        else:
                            row_data.append(str(float(element.evalf())))
                    f.write(",".join(row_data) + "\n")
        
        Messagebox.show_info( f"{title}已导出到: {file_path}")
        
    except Exception as e:
        Messagebox.show_error(f"导出{title}失败: {str(e)}")

def export_matrix_A():
    ensureA()
    export_matrix_data(Amatrix, "矩阵A")

def export_matrix_B():
    ensureB()
    export_matrix_data(Bmatrix, "矩阵B")

def export_result():
    """导出计算结果矩阵"""
    if C_matrix is None and current_eigenvals is None and current_eigenvects is None:
        Messagebox.show_error("没有计算结果可导出")
        return

    file_path = filedialog.asksaveasfilename(
        title="保存结果",
        defaultextension=".txt",
        filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv")]
    )

    if not file_path:
        return

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"矩阵计算器导出结果\n")
            f.write(f"操作类型: {comboC.get()}\n")
            f.write("="*50 + "\n\n")

            if comboC.get() == "A的特征值" and current_eigenvals is not None:
                f.write("特征值:\n")
                for val, mult in current_eigenvals.items():
                    f.write(f"特征值: {val}, 重数: {mult}\n")

            elif comboC.get() == "A的特征向量" and current_eigenvects is not None:
                f.write("特征值和特征向量:\n")
                for val, mult, vects in current_eigenvects:
                    f.write(f"\n特征值: {val}\n")
                    for i, vect in enumerate(vects):
                        f.write(f"特征向量 {i+1}: ")
                        for element in vect:
                            f.write(f"{element} ")
                        f.write("\n")

            elif comboC.get() in ("LU分解", "QR分解", "奇异值分解"):
                f.write("分解结果:\n")
                if comboC.get() == "LU分解":
                    L, U, _ = C_matrix  # C_matrix 是 (L, U, P)
                    f.write("L矩阵:\n")
                    for i in range(L.rows):
                        f.write("\t".join(str(L[i, j]) for j in range(L.cols)) + "\n")
                    f.write("\nU矩阵:\n")
                    for i in range(U.rows):
                        f.write("\t".join(str(U[i, j]) for j in range(U.cols)) + "\n")

                elif comboC.get() == "QR分解":
                    Q, R = C_matrix
                    f.write("Q矩阵:\n")
                    for i in range(Q.rows):
                        f.write("\t".join(str(Q[i, j]) for j in range(Q.cols)) + "\n")
                    f.write("\nR矩阵:\n")
                    for i in range(R.rows):
                        f.write("\t".join(str(R[i, j]) for j in range(R.cols)) + "\n")

                elif comboC.get() == "奇异值分解":
                    U, S, V = C_matrix
                    f.write("U矩阵:\n")
                    for i in range(U.rows):
                        f.write("\t".join(str(U[i, j]) for j in range(U.cols)) + "\n")
                    f.write("\nΣ（奇异值对角矩阵）:\n")
                    for i in range(S.rows):
                        f.write("\t".join(str(S[i, j]) for j in range(S.cols)) + "\n")
                    f.write("\nV转置矩阵:\n")
                    for i in range(V.rows):
                        f.write("\t".join(str(V[i, j]) for j in range(V.cols)) + "\n")

            elif C_matrix is not None:
                f.write("结果矩阵:\n")
                for i in range(C_matrix.rows):
                    f.write("\t".join(str(C_matrix[i, j]) for j in range(C_matrix.cols)) + "\n")

            # 如果是CSV格式且是可导出的矩阵
            if file_path.endswith('.csv') and C_matrix is not None:
                f.write("\n\nCSV格式:\n")
                for i in range(C_matrix.rows):
                    row_data = []
                    for j in range(C_matrix.cols):
                        element = C_matrix[i, j]
                        if element.has(sp.I):
                            row_data.append(str(element))
                        else:
                            row_data.append(str(float(element.evalf())))
                    f.write(",".join(row_data) + "\n")

        Messagebox.show_info(f"结果已导出到: {file_path}")

    except Exception as e:
        Messagebox.show_error(f"导出失败: {str(e)}")

def export_plot():
    """导出可视化图片"""
    file_path = filedialog.asksaveasfilename(
        title="保存图片",
        defaultextension=".png",
        filetypes=[("PNG图片", "*.png"), ("JPG图片", "*.jpg"), ("PDF文件", "*.pdf"), ("SVG文件", "*.svg")]
    )
    if file_path:
        try:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            Messagebox.show_info( f"图片已保存到: {file_path}")
        except Exception as e:
            Messagebox.show_error(f"保存图片失败: {str(e)}")

def export_3d_plot():
    """导出3D特征向量图片"""
    file_path = filedialog.asksaveasfilename(
        title="保存3D图片",
        defaultextension=".png",
        filetypes=[("PNG图片", "*.png"), ("JPG图片", "*.jpg"), ("PDF文件", "*.pdf")]
    )
    if file_path:
        try:
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            Messagebox.show_info( f"3D图片已保存到: {file_path}")
        except Exception as e:
            Messagebox.show_error(f"保存图片失败: {str(e)}")

# ======================================
# 可视化功能
# ======================================
def visualize_matrix_2d(matrix, title="矩阵可视化"):
    """2D柱状图可视化矩阵"""
    lazy_imports()
    if matrix is None:
        return 
    np_matrix = matrix_to_numpy(matrix)
    if np_matrix is None:
        Messagebox.show_error("无法转换矩阵")
        return
    
    global fig
    
    # 创建新窗口
    viz_window = ttk.Toplevel(root)
    viz_window.title(title)
    viz_window.geometry("")  # 添加这行强制重新计算窗口大小
    
    def refresh_plot():
        plt.clf()
        viz_window.destroy()
        visualize_matrix_2d(matrix, title)
        
    # 导出按钮框架 - 固定在顶部，增加底部边距
    export_frame = ttk.Frame(viz_window)
    export_frame.pack(side=TOP, fill=X, padx=10, pady=(5, 15))  # 增加底部边距
    ttk.Button(export_frame, text="导出图片", command=export_plot, bootstyle=(INFO, OUTLINE)).pack(side=LEFT, padx=5)
    ttk.Button(export_frame, text="刷新图像", command=refresh_plot, bootstyle=(WARNING, OUTLINE)).pack(side=LEFT, padx=5)
    
    # 图表框架 - 填充剩余空间
    plot_frame = ttk.Frame(viz_window)
    plot_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=(0, 10))
    
    fig = plt.figure(figsize=(9, 6))  # 进一步调整图表大小
    
    rows, cols = np_matrix.shape
    
    if rows <= 10 and cols <= 10:
        # 小矩阵：显示具体数值的2D柱状图
        ax = fig.add_subplot(111, projection='3d')
        
        x_pos, y_pos = np.meshgrid(range(cols), range(rows))
        x_pos = x_pos.flatten()
        y_pos = y_pos.flatten()
        z_pos = np.zeros_like(x_pos)
        
        # 获取矩阵值（如果是复数，取模）
        if np.iscomplexobj(np_matrix):
            heights = np.abs(np_matrix).flatten()
        else:
            heights = np_matrix.flatten()
        
        dx = dy = 0.8
        dz = heights
        
        # 根据值的大小设置颜色
        colors = plt.cm.viridis(heights / (np.max(heights) if np.max(heights) != 0 else 1))
        
        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors, alpha=0.8)
        
        # 添加数值标签
        for i in range(rows):
            for j in range(cols):
                value = np_matrix[i, j]
                if np.iscomplexobj(np_matrix) and np.imag(value) != 0:
                    label = f"{value:.2f}"
                else:
                    label = f"{np.real(value):.2f}"
                ax.text(j, i, heights[i*cols + j] + 0.1, label, 
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        ax.set_zlabel('值')
        ax.set_title(f'{title} ({rows}×{cols})')
        
    else:
        # 大矩阵：热力图形式
        ax = fig.add_subplot(111)
        
        # 如果是复数矩阵，显示模
        if np.iscomplexobj(np_matrix):
            plot_matrix = np.abs(np_matrix)
            ax.set_title(f'{title} (模) ({rows}×{cols})')
        else:
            plot_matrix = np_matrix
            ax.set_title(f'{title} ({rows}×{cols})')
        
        im = ax.imshow(plot_matrix, cmap='viridis', aspect='auto')
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 对于中等大小的矩阵，显示网格
        if rows <= 20 and cols <= 20:
            ax.set_xticks(range(cols))
            ax.set_yticks(range(rows))
            ax.grid(True, alpha=0.3)
    
    # 调整布局，为顶部标题留出更多空间
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    
    # 将canvas嵌入到plot_frame而不是viz_window
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

def visualize_matrix_A():
    ensureA()
    visualize_matrix_2d(Amatrix, "矩阵A可视化")

def visualize_matrix_B():
    ensureB()
    visualize_matrix_2d(Bmatrix, "矩阵B可视化")

def visualize_2d_eigenvectors():
    """2D可视化2×2矩阵的特征向量"""
    lazy_imports()
    if current_eigenvects is None:
        Messagebox.show_error("请先计算特征向量")
        return

    global fig

    # 创建新窗口
    viz_window = ttk.Toplevel(root)
    viz_window.title("2×2矩阵特征向量2D可视化")
    viz_window.geometry("")  # 稍微增加高度给按钮留空间
    
    def refresh_plot():
        plt.clf()
        viz_window.destroy()
        visualize_matrix_2d(matrix, title)

    # 导出按钮框架 - 固定在顶部
    export_frame = ttk.Frame(viz_window)
    export_frame.pack(side=TOP, fill=X, padx=10, pady=5)
    ttk.Button(export_frame, text="导出图片", command=export_plot, bootstyle=(INFO, OUTLINE)).pack(side=LEFT, padx=5)
    ttk.Button(export_frame, text="刷新图像", command=refresh_plot, bootstyle=(WARNING, OUTLINE)).pack(side=LEFT, padx=5)

    # 图表框架 - 填充剩余空间
    plot_frame = ttk.Frame(viz_window)
    plot_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=5)

    fig, ax = plt.subplots(figsize=(8, 7))  # 稍微调整图表大小

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_idx = 0

    # 绘制坐标轴
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.axvline(0, color='black', lw=1, alpha=0.5)
    ax.set_xlabel("X轴")
    ax.set_ylabel("Y轴")
    ax.set_title("2×2矩阵特征向量2D可视化")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)

    # 绘制每个特征向量及其反向
    for eigenval, mult, eigenvects in current_eigenvects:
        for vect in eigenvects:
            try:
                v = [float(vect[0].evalf()), float(vect[1].evalf())]
            except:
                v = [float(complex(vect[0].evalf()).real), float(complex(vect[1].evalf()).real)]

            norm = np.linalg.norm(v)
            if norm > 0:
                v = [x / norm * 2 for x in v]  # 缩放长度为2单位

            color = colors[color_idx % len(colors)]
            label = f"特征值 {format_sympy_output(eigenval)}"
            ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color=color, linewidth=2, label=label)
            ax.quiver(0, 0, -v[0], -v[1], angles='xy', scale_units='xy', scale=1, color=color, alpha=0.5, linestyle='dashed')
            color_idx += 1

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend()
    plt.tight_layout()

    # 将canvas嵌入到plot_frame而不是viz_window
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)


def visualize_3d_eigenvectors():
    """3D可视化3×3矩阵的特征向量"""
    lazy_imports()
    if current_eigenvects is None:
        Messagebox.show_error("请先计算特征向量")
        return
    
    global fig
    
    # 创建新窗口
    viz_window = ttk.Toplevel(root)
    viz_window.title("3×3矩阵特征向量3D可视化")
    viz_window.geometry("")  # 稍微增加高度给按钮留空间
    
    def refresh_plot():
        plt.clf()
        viz_window.destroy()
        visualize_matrix_2d(matrix, title)
    
    # 导出按钮框架 - 固定在顶部
    export_frame = ttk.Frame(viz_window)
    export_frame.pack(side=TOP, fill=X, padx=10, pady=5)
    ttk.Button(export_frame, text="导出3D图片", command=export_3d_plot, bootstyle=(INFO, OUTLINE)).pack(side=LEFT, padx=5)
    ttk.Button(export_frame, text="刷新图像", command=refresh_plot, bootstyle=(WARNING, OUTLINE)).pack(side=LEFT, padx=5)
    
    # 图表框架 - 填充剩余空间
    plot_frame = ttk.Frame(viz_window)
    plot_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=5)
    
    fig = plt.figure(figsize=(10, 7))  # 稍微调整图表大小
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    color_idx = 0
    
    # 绘制坐标轴
    ax.quiver(0, 0, 0, 2, 0, 0, color='black', alpha=0.3, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 2, 0, color='black', alpha=0.3, arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 2, color='black', alpha=0.3, arrow_length_ratio=0.1)
    ax.text(2.2, 0, 0, 'X', fontsize=12)
    ax.text(0, 2.2, 0, 'Y', fontsize=12)
    ax.text(0, 0, 2.2, 'Z', fontsize=12)
    
    # 绘制每个特征向量
    for eigenval, mult, eigenvects in current_eigenvects:
        for i, eigenvect in enumerate(eigenvects):
            # 转换特征向量为浮点数
            try:
                v = [float(eigenvect[j].evalf()) for j in range(3)]
            except:
                # 如果有复数，取实部
                v = [float(complex(eigenvect[j].evalf()).real) for j in range(3)]
            
            # 归一化向量（可选）
            norm = np.linalg.norm(v)
            if norm > 0:
                v = [x/norm * 2 for x in v]  # 缩放到长度2
            
            # 绘制特征向量
            ax.quiver(0, 0, 0, v[0], v[1], v[2], 
                     color=colors[color_idx % len(colors)], 
                     arrow_length_ratio=0.1, 
                     linewidth=3,
                     label=f'特征值 {float(eigenval.evalf()):.3f}')
            
            # 绘制反向向量（显示向量的双向性）
            ax.quiver(0, 0, 0, -v[0], -v[1], -v[2], 
                     color=colors[color_idx % len(colors)], 
                     arrow_length_ratio=0.1, 
                     linewidth=2, 
                     alpha=0.6)
            
            color_idx += 1
    
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')
    ax.set_title('3×3矩阵特征向量3D可视化')
    ax.legend()
    
    # 设置坐标轴范围
    max_range = 2.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    
    # 将canvas嵌入到plot_frame而不是viz_window
    canvas = FigureCanvasTkAgg(fig, plot_frame)
    canvas.get_tk_widget().pack(fill=BOTH, expand=True)

def visualize_submatrices_3d(C_matrix, base_title="分解结果"):
    if not isinstance(C_matrix, tuple):
        Messagebox.show_error("当前不是分解结果")
        return
    
    for idx, mat in enumerate(C_matrix):
        title = f"{base_title} - 子矩阵 {idx+1}"
        visualize_matrix_2d(mat, title)

def visualize_result():
    """通用可视化入口，判断是矩阵还是分解结果"""
    if comboC.get() == "A的特征向量" and current_eigenvects is not None:
        # 特征向量可视化
        if Amatrix.rows == 3 and Amatrix.cols == 3:
            visualize_3d_eigenvectors()
        elif Amatrix.rows == 2 and Amatrix.cols == 2:
            visualize_2d_eigenvectors()
        else:
            Messagebox.show_info("目前只支持 2×2 和 3×3 矩阵的特征向量可视化")
    elif isinstance(C_matrix, tuple):
        # 分解结果的多个子矩阵 → 逐个3D可视化
        visualize_submatrices_3d(C_matrix, "分解结果")
    elif C_matrix is not None:
        # 普通矩阵结果可视化
        visualize_matrix_2d(C_matrix, "计算结果可视化")
    else:
        Messagebox.show_error("没有可视化的结果")

# ======================================
# 历史记录功能
# ======================================
def save_to_history():
    """保存当前计算到历史记录"""
    lazy_imports()
    global calculation_history, Amatrix, Bmatrix, Bnum, C_matrix
    
    if Amatrix is None:
        return
    
    # 创建历史记录条目
    history_entry = {
        'timestamp': arrow.now().format('YYYY-MM-DD HH:mm:ss'),
        'operation': comboC.get(),
        'matrix_A': Amatrix.copy() if Amatrix is not None else None,
        'matrix_B': Bmatrix.copy() if Bmatrix is not None else None,
        'number_B': Bnum,
        'result': tuple(mat.copy() for mat in C_matrix) if isinstance(C_matrix, tuple) else (C_matrix.copy() if C_matrix is not None else None),
        'scalar_result': current_scalar_result,
        'eigenvals': current_eigenvals.copy() if current_eigenvals is not None else None,
        'eigenvects': current_eigenvects.copy() if current_eigenvects is not None else None
    }
    
    calculation_history.append(history_entry)
    
    # 限制历史记录数量（最多保存50条）
    if len(calculation_history) > 50:
        calculation_history.pop(0)

def show_calculation_history():
    """显示计算历史记录窗口"""
    global calculation_history
    
    if not calculation_history:
        Messagebox.show_info("暂无计算历史记录")
        return
    
    # 创建历史记录窗口
    history_window = ttk.Toplevel(root)
    history_window.title("计算历史记录")
    history_window.geometry("1500x900")
    history_window.resizable(True, True)
    
    # 创建主框架
    main_frame = ttk.Frame(history_window)
    main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    # 创建左侧历史列表框架
    left_frame = ttk.LabelFrame(main_frame, text="历史记录列表", bootstyle='info')
    left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 5))
    
    # 创建列表框和滚动条
    listbox_frame = ttk.Frame(left_frame)
    listbox_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    history_listbox = tk.Listbox(listbox_frame, height=20, font=("黑体", 10))
    scrollbar = ttk.Scrollbar(listbox_frame, orient=VERTICAL, command=history_listbox.yview)
    history_listbox.config(yscrollcommand=scrollbar.set)
    
    history_listbox.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    # 填充历史记录列表
    for i, entry in enumerate(reversed(calculation_history)):  # 最新的在前面
        list_text = f"{entry['timestamp']} - {entry['operation']}"
        if entry['matrix_A'] is not None:
            list_text += f" [A:{entry['matrix_A'].rows}×{entry['matrix_A'].cols}]"
        if entry['matrix_B'] is not None:
            list_text += f" [B:{entry['matrix_B'].rows}×{entry['matrix_B'].cols}]"
        elif entry['number_B'] is not None:
            list_text += f" [b:{entry['number_B']}]"
        
        history_listbox.insert(END, list_text)
    
    # 创建右侧详情框架
    right_frame = ttk.LabelFrame(main_frame, text="详细信息", bootstyle='success')
    right_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=(5, 0))
    
    # 详情显示文本框
    detail_text = ScrolledText(right_frame, height=25, width=50, font=("Consolas", 10))
    detail_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    # 底部按钮框架
    button_frame = ttk.Frame(right_frame)
    button_frame.pack(fill=X, padx=5, pady=5)
    
    def on_history_select(event):
        """当选择历史记录时显示详情"""
        selection = history_listbox.curselection()
        if not selection:
            return
        
        # 获取选中的历史记录（注意索引反转）
        index = len(calculation_history) - 1 - selection[0]
        entry = calculation_history[index]
        
        # 清空详情文本框
        detail_text.delete(1.0, END)
        
        # 显示详细信息
        detail_text.insert(END, f"计算时间: {entry['timestamp']}\n")
        detail_text.insert(END, f"操作类型: {entry['operation']}\n")
        detail_text.insert(END, "="*50 + "\n\n")
        
        # 显示矩阵A
        if entry['matrix_A'] is not None:
            detail_text.insert(END, f"矩阵A ({entry['matrix_A'].rows}×{entry['matrix_A'].cols}):\n")
            for i in range(entry['matrix_A'].rows):
                for j in range(entry['matrix_A'].cols):
                    element = entry['matrix_A'][i, j]
                    detail_text.insert(END, f"{format_sympy_output(element)} ")
                detail_text.insert(END, "\n")
            detail_text.insert(END, "\n")
        
        # 显示矩阵B或数值b
        if entry['matrix_B'] is not None:
            detail_text.insert(END, f"矩阵B ({entry['matrix_B'].rows}×{entry['matrix_B'].cols}):\n")
            for i in range(entry['matrix_B'].rows):
                for j in range(entry['matrix_B'].cols):
                    element = entry['matrix_B'][i, j]
                    detail_text.insert(END, f"{format_sympy_output(element)} ")
                detail_text.insert(END, "\n")
            detail_text.insert(END, "\n")
        elif entry['number_B'] is not None:
            detail_text.insert(END, f"数值b: {entry['number_B']}\n\n")
        
        # 显示计算结果
        detail_text.insert(END, "计算结果:\n")
        if entry['result'] is not None:
            if isinstance(entry['result'], tuple):
                for idx, mat in enumerate(entry['result']):
                    if mat is not None:
                        detail_text.insert(END, f"子矩阵 {idx+1} ({mat.rows}×{mat.cols}):\n")
                        for i in range(mat.rows):
                            for j in range(mat.cols):
                                element = mat[i, j]
                                detail_text.insert(END, f"{format_sympy_output(element)} ")
                            detail_text.insert(END, "\n")
                        detail_text.insert(END, "\n")
            else:
                mat = entry['result']
                detail_text.insert(END, f"结果矩阵 ({mat.rows}×{mat.cols}):\n")
                for i in range(mat.rows):
                    for j in range(mat.cols):
                        element = mat[i, j]
                        detail_text.insert(END, f"{format_sympy_output(element)} ")
                    detail_text.insert(END, "\n")
                
        elif entry['eigenvals'] is not None:
            for val, mult in entry['eigenvals'].items():
                detail_text.insert(END, f"特征值: {format_sympy_output(val)}, 重数: {mult}\n")
        elif entry['eigenvects'] is not None:
            for val, mult, vects in entry['eigenvects']:
                detail_text.insert(END, f"特征值: {format_sympy_output(val)}, 重数: {mult}\n")
                for i, vect in enumerate(vects):
                    detail_text.insert(END, f"特征向量{i+1}: ")
                    for element in vect:
                        detail_text.insert(END, f"{format_sympy_output(element)} ")
                    detail_text.insert(END, "\n")
                detail_text.insert(END, "\n")
        elif 'scalar_result' in entry and entry['scalar_result'] is not None:
            detail_text.insert(END, f"{entry['operation']}结果: {format_sympy_output(entry['scalar_result'])}\n")

    
    
    def load_history_data():
        """将选中的历史记录加载到主界面"""
        selection = history_listbox.curselection()
        if not selection:
            Messagebox.show_warning("请先选择一条历史记录")
            return
        
        global Amatrix, Bmatrix, Bnum
        
        # 获取选中的历史记录
        index = len(calculation_history) - 1 - selection[0]
        entry = calculation_history[index]
        
        # 加载矩阵A
        if entry['matrix_A'] is not None:
            Amatrix = entry['matrix_A'].copy()
            textA.delete(1.0, END)
            for i in range(Amatrix.rows):
                for j in range(Amatrix.cols):
                    element = Amatrix[i, j]
                    textA.insert(END, f"{format_sympy_output(element)} ")
                textA.insert(END, "\n")
        
        # 加载矩阵B或数值b
        if entry['matrix_B'] is not None:
            Bmatrix = entry['matrix_B'].copy()
            textB.delete(1.0, END)
            for i in range(Bmatrix.rows):
                for j in range(Bmatrix.cols):
                    element = Bmatrix[i, j]
                    textB.insert(END, f"{format_sympy_output(element)} ")
                textB.insert(END, "\n")
        elif entry['number_B'] is not None:
            Bnum = entry['number_B']
            Bmatrix = None
            textB.delete(1.0, END)
            textB.insert(1.0, str(entry['number_B']))
        
        # 设置操作类型
        operation_list = list(comboC['values'])
        if entry['operation'] in operation_list:
            comboC.current(operation_list.index(entry['operation']))
        
        Messagebox.show_info("历史数据已加载到主界面")
        history_window.destroy()
    
    def export_history():
        """导出历史记录"""
        if not calculation_history:
            Messagebox.show_warning("没有历史记录可导出")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="导出历史记录",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("CSV文件", "*.csv")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("矩阵计算器历史记录\n")
                f.write("="*80 + "\n\n")
                
                for i, entry in enumerate(calculation_history, 1):
                    f.write(f"记录 {i}:\n")
                    f.write(f"时间: {entry['timestamp']}\n")
                    f.write(f"操作: {entry['operation']}\n")
                    
                    if entry['matrix_A'] is not None:
                        f.write(f"矩阵A ({entry['matrix_A'].rows}×{entry['matrix_A'].cols}):\n")
                        for row_i in range(entry['matrix_A'].rows):
                            for col_j in range(entry['matrix_A'].cols):
                                f.write(f"{entry['matrix_A'][row_i, col_j]}\t")
                            f.write("\n")
                        f.write("\n")
                    
                    if entry['matrix_B'] is not None:
                        f.write(f"矩阵B/数b ({entry['matrix_B'].rows}×{entry['matrix_B'].cols}):\n")
                        for row_i in range(entry['matrix_B'].rows):
                            for col_j in range(entry['matrix_B'].cols):
                                f.write(f"{entry['matrix_B'][row_i, col_j]}\t")
                            f.write("\n")
                        f.write("\n")
                    elif entry['number_B'] is not None:
                        f.write(f"数值b: {entry['number_B']}\n\n")
                    
                    f.write("-"*60 + "\n\n")
            
            Messagebox.show_info(f"历史记录已导出到: {file_path}")
            
        except Exception as e:
            Messagebox.show_error(f"导出历史记录失败: {str(e)}")
    
    def clear_history():
        """清空历史记录"""
        result = Messagebox.show_question("确认要清空所有历史记录吗？此操作不可撤销。")
        if result == "Yes":
            global calculation_history
            calculation_history.clear()
            history_listbox.delete(0, END)
            detail_text.delete(1.0, END)
            Messagebox.show_info("历史记录已清空")
    
    # 绑定列表选择事件
    history_listbox.bind('<<ListboxSelect>>', on_history_select)
    
    # 添加按钮
    ttk.Button(button_frame, text="加载到主界面", command=load_history_data, 
              bootstyle=(SUCCESS, OUTLINE), width=12).pack(side=LEFT, padx=2)
    ttk.Button(button_frame, text="导出历史", command=export_history, 
              bootstyle=(INFO, OUTLINE), width=12).pack(side=LEFT, padx=2)
    ttk.Button(button_frame, text="清空历史", command=clear_history, 
              bootstyle=(DANGER, OUTLINE), width=12).pack(side=LEFT, padx=2)
    ttk.Button(button_frame, text="关闭", command=history_window.destroy, 
              bootstyle=(SECONDARY, OUTLINE), width=12).pack(side=RIGHT, padx=2)

# ======================================
# GUI 布局和主程序

def lazy_imports():
    global plt, FigureCanvasTkAgg, arrow, filedialog
    if plt is None:
        import matplotlib.pyplot as _plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _Canvas
        _plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        _plt.rcParams['axes.unicode_minus'] = False
        plt = _plt
        FigureCanvasTkAgg = _Canvas
    if arrow is None:
        import arrow as _arrow
        arrow = _arrow
    if filedialog is None:
        from tkinter import filedialog as _filedialog
        filedialog = _filedialog


# ======================================
# 如果是打包环境，就手动添加 .exe 所在路径到 DLL 搜索路径
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
    os.environ['PATH'] = base_dir + os.pathsep + os.environ.get('PATH', '')

def resource_path(relative_path):
    """获取资源文件的绝对路径，适用于开发环境和PyInstaller打包后的环境"""
    try:
        # PyInstaller创建临时文件夹，并将路径存储在_MEIPASS中
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境中使用当前脚本所在目录
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

def set_application_icon(window, icon_filename):
    """设置应用程序图标"""
    try:
        icon_path = resource_path(icon_filename)
        if os.path.exists(icon_path):
            # 设置窗口和任务栏图标
            window.iconbitmap(icon_path)
            window.wm_iconbitmap(icon_path)
            print(f"图标设置成功: {icon_path}")
        else:
            print(f"图标文件不存在: {icon_path}")
    except Exception as e:
        print(f"设置图标失败: {e}")

def on_closing():
    """改进的窗口关闭处理"""
    try:
        # 设置关闭标志
        shutdown_event.set()
        
        # 等待所有子线程结束
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join(timeout=1.0)
        
        # 清理GUI资源
        if 'root' in globals():
            root.quit()
            root.destroy()
            
    except Exception as e:
        print(f"关闭时发生错误: {e}")
    finally:
        # 强制退出
        os._exit(0)
        
def main():
    global root, EntryA, EntryB, textA, textB, textC, comboB, comboC, matrix_value, matrix_value_dist, entryA, entryB, spinboxA, spinboxB
    root = ttk.Window(title="矩阵计算器",        #设置窗口的标题
            themename="superhero",     #设置主题
            size=(1500,900))
    
    # 添加窗口关闭协议
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 设置程序图标 - 适用于打包后的程序
    set_application_icon(root, "function.ico")  # 确保icon.ico文件在项目根目录
    
    root.resizable(width=False,height=False)
    
    FR = ttk.LabelFrame(root, text="控制面板",bootstyle='warning')
    FR.place(x=10, y=6, width=1480, height=288)
    
    # 导入矩阵A
    ttk.Button(FR, text="导入矩阵A", command=A,bootstyle=(INFO,OUTLINE)).place(x=50, y=18)
    ttk.Label(FR, text="文件位置:", bootstyle=INFO).place(x=200, y=21)
    EntryA = ttk.Entry(FR, width=36)
    EntryA.place(x=300, y=21)
    
    # 导入矩阵B
    ttk.Button(FR, text="导入矩阵B", command=B,bootstyle=(INFO,OUTLINE)).place(x=790, y=18)
    ttk.Label(FR, text="文件位置:", bootstyle=INFO).place(x=940, y=21)
    EntryB = ttk.Entry(FR, width=36)
    EntryB.place(x=1040, y=21)
    
    # 矩阵生成区域
    fr = ttk.LabelFrame(FR, text="矩阵生成",bootstyle='success')
    fr.place(x=10, y=72, width=1460, height=96)
    
    ttk.Label(fr, text="目标矩阵:", bootstyle=INFO).place(x=50, y=15)
    matrix_value = ttk.StringVar()
    matrix_value.set(0)
    matrix_value_dist = {"0":"矩阵A","1":"矩阵B",}
    ttk.Radiobutton(fr, text='矩阵A', variable=matrix_value, value=0).place(x=140, y=17)
    ttk.Radiobutton(fr, text='矩阵B', variable=matrix_value, value=1).place(x=230, y=17)
    
    ttk.Label(fr, text="分布类型:", bootstyle=INFO).place(x=330, y=15)
    comboB = ttk.Combobox(fr, values=("正态分布", "均匀分布", "泊松分布", "指数分布"), 
                         font=("黑体", 10), state="readonly", width=10)
    comboB.current(0)
    comboB.place(x=430, y=16)
    
    ttk.Label(fr, text="参数1：",bootstyle=INFO).place(x=600, y=15)
    entryA = ttk.Entry(fr, width=5)
    entryA.insert(0, "1.0")
    entryA.place(x=660, y=15)
    
    ttk.Label(fr, text="参数2：", bootstyle=INFO).place(x=740, y=15)
    entryB = ttk.Entry(fr, width=5)
    entryB.insert(0, "1.0")
    entryB.place(x=810, y=15)
    
    ttk.Label(fr, text="行：", bootstyle=INFO).place(x=895, y=15)
    spinboxA = ttk.Spinbox(fr, width=5, from_=1, to=10)
    spinboxA.insert(0, "1")
    spinboxA.place(x=930, y=15)
    
    ttk.Label(fr, text="列：", bootstyle=INFO).place(x=1040, y=15)
    spinboxB = ttk.Spinbox(fr, width=5, from_=1, to=10)
    spinboxB.insert(0, "1")
    spinboxB.place(x=1080, y=15)
    
    ttk.Button(fr, text="生成", command=creat, width=10,bootstyle=(SUCCESS,OUTLINE)).place(x=1210, y=15)
    
    # 操作选择
    ttk.Label(FR, text="选择操作:", bootstyle=INFO).place(x=50, y=198)
    comboC = ttk.Combobox(FR, values=("加法A+B", "减法A-B", "数乘bxA", "乘法AxB", "A的秩", "A的转置", "A的行列式", "A的逆矩阵", "A的伴随矩阵", "幂A^b", "A的特征值", "A的特征向量",
                                      "求解AX=B","交换A的1和b行","交换A的1和b列","A的范数","A的迹","A的共轭转置","水平拼接","垂直拼接", 
                                      "A的LU分解", "A的QR分解", "A的奇异值分解"), 
                         font=("黑体", 10), state="readonly", width=15)
    comboC.current(0)
    comboC.place(x=140, y=199)
    
    # 操作按钮
    ttk.Button(FR, text="执行计算", command=operation, width=10,bootstyle=(INFO,OUTLINE)).place(x=370, y=198)
    ttk.Button(FR, text="导出结果", command=export_result, width=10,bootstyle=(INFO,OUTLINE)).place(x=540, y=198)
    
    # 主题选择
    style = ttk.Style()
    theme_names = style.theme_names()
    
    def change_theme(e):
        t = theme_cbo.get()
        style.theme_use(t)
        theme_cbo.selection_clear()
    
    lbl = ttk.Label(FR, text="选择主题:")
    theme_cbo = ttk.Combobox(
        master=FR,
        text=style.theme.name,
        values=theme_names,
        state="readonly"
    )
    theme_cbo.place(x=840, y=199)
    theme_cbo.current(theme_names.index(style.theme.name))
    lbl.place(x=740, y=198)
    
    theme_cbo.bind('<<ComboboxSelected>>', change_theme)
    
    # 历史记录
    ttk.Button(FR, text="历史记录", command=show_calculation_history, width=10,bootstyle=(INFO,OUTLINE)).place(x=1290, y=198)
    
    # 矩阵A区域
    fr1 = ttk.LabelFrame(root, text="矩阵A", width=730, height=288,bootstyle='secondary')
    fr1.place(x=10, y=306)
    textA = ttk.Text(fr1, width=63, height=8)
    textA.place(x=10, y=6)
    
    ttk.Button(fr1, text="可视化", command=visualize_matrix_A, width=10,bootstyle=(SECONDARY,OUTLINE)).place(x=30, y=216)
    ttk.Button(fr1, text="导出矩阵", command=export_matrix_A, width=10,bootstyle=(SECONDARY,OUTLINE)).place(x=580, y=216)
    
    # 矩阵B区域
    fr2 = ttk.LabelFrame(root, text="矩阵B/数b", width=730, height=288,bootstyle='secondary')
    fr2.place(x=760, y=306)
    textB = ttk.Text(fr2, width=63, height=8)
    textB.place(x=10, y=6)
    
    ttk.Button(fr2, text="可视化", command=visualize_matrix_B, width=10,bootstyle=(SECONDARY,OUTLINE)).place(x=30, y=216)
    ttk.Button(fr2, text="导出矩阵", command=export_matrix_B, width=10,bootstyle=(SECONDARY,OUTLINE)).place(x=580, y=216)
    
    # 结果区域
    fr3 = ttk.LabelFrame(root, text="计算结果",bootstyle='danger')
    fr3.place(x=10, y=606, width=1480, height=288)
    textC = ttk.Text(fr3, width=131, height=8)
    textC.config(state="disabled")
    textC.place(x=10, y=6)
    
    ttk.Button(fr3, text="结果可视化", command=visualize_result, width=10,bootstyle=(DANGER,OUTLINE)).place(x=675, y=216)
    
    root.mainloop()


if __name__ == '__main__':
    main()
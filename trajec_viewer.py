import vispy
from vispy import scene
from vispy.scene import visuals
from vispy.scene.visuals import GridLines
from vispy.scene.visuals import XYZAxis
from vispy.scene.visuals import LinePlot
from vispy.color import ColorArray

from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QDialog
from PyQt5.QtWidgets import QPushButton, QSlider, QLabel, QFileDialog, QProgressBar
from PyQt5.QtCore import QTimer
from PyQt5 import uic

from scipy.interpolate import interp1d
import numpy as np
import quaternion
import pandas as pd
import json
import os


class MyUi(QMainWindow):
    def __init__(self):
        super(MyUi, self).__init__()
        uic.loadUi('viewer.ui', self)

        self.vispy_view = self.findChild(QVBoxLayout, 'vispy_view')

        self.pause_btn = self.findChild(QPushButton, 'pause_button')
        self.start_btn = self.findChild(QPushButton, 'start_button')
        self.stop_btn = self.findChild(QPushButton, 'stop_button')

        # slider
        self.t_slider = self.findChild(QSlider, 'time_slider')
        # progress bar
        self.progress_bar = self.findChild(QProgressBar, 'progressBar')

        # load buttons
        self.load_param_title = self.findChild(QLabel, 'param_load_title')
        self.load_param_btn = self.findChild(QPushButton, 'param_load_button')
        self.load_trajec_title = self.findChild(QLabel, 'trajec_load_title')
        self.load_trajec_btn = self.findChild(QPushButton, 'trajec_load_button')
        self.load_obj_title = self.findChild(QLabel, 'model_load_title')
        self.load_obj_btn = self.findChild(QPushButton, 'model_load_button')
        self.import_btn = self.findChild(QPushButton, 'load_button')

        # status
        self.t_text = self.findChild(QLabel, 't_val')
        self.v_text = self.findChild(QLabel, 'v_val')
        self.r_text = self.findChild(QLabel, 'r_val')
        self.w_text = self.findChild(QLabel, 'w_val')

    def show_file_dialog(self, title='Open File', path='./', ext='*.*'):
        fname = QFileDialog.getOpenFileName(self, title, path, ext)
        return fname[0] # return file path


class RocketMesh:
    def __init__(self, filename, CG_pos=0.0):
        self.CG_pos = np.array([CG_pos, 0.0, 0.0])

        v, f, n, t = vispy.io.read_mesh(filename)
        # self.vertices = -np.array(v)/1000.0 # XXX: スケーリングと初期角度調整
        self.vertices = np.array(v)
        self.faces = np.array(f)
        self.normals = np.array(n)
        self.texcoords = np.array(t)
        self.visual = visuals.Mesh(self.vertices, self.faces, color='red')
        self.visual.light_dir = [-0.3, -0.3, -1.0]
        # self.visual.ambient_color = 'gray'

        # self.pos = np.zeros((3))
        # self.q = np.zeros((4))
    
    def set_scale(self, scale):
        '''
        scale: scale array [sx, sy, sz]
        '''
        self.vertices *= scale

    def set_CG_pos(self, pos):
        self.CG_pos = pos
        # self.move(self.pos, self.q)
    
    def set_vertices(self, v):
        self.visual.set_data(v, self.faces, color='red')
        self.visual.light_dir = [-0.3, -0.3, -1.0]

    def move(self, pos, q): # q: float*4 array of attitude quaternion
        _v = np.copy(self.vertices) + self.CG_pos
        
        _q = quaternion.as_quat_array(q)
        attitude = quaternion.as_rotation_matrix(_q)

        _v = np.dot(attitude, _v.T).T
        _v += pos

        # self.pos = np.copy(pos)
        # self.q = np.copy(q)
        self.visual.set_data(_v, self.faces, color='red')
        self.visual.light_dir = [-0.3, -0.3, -1.0]
        # self.visual.ambient_color = 'gray'


class UIHandler:
    def __init__(self, ui:MyUi, canvas, camera):
        self.ui = ui
        self.canvas = canvas
        self.camera = camera
        self.ui.vispy_view.addWidget(canvas.native)

        # self._trajec_loaded = False
        # self._param_loaded = False

        # member variables initialization
        self._ready = False
        self._slider_dt = 0.01
        self._playback_mode = False
        self._playback_t = 0.0
        self._playback_timer = QTimer(self.ui)
        self.obj_file = ''
        self.param_file = ''
        self.trajec_file = ''

        # self.rocket_model = RocketMesh('bianca.obj')

        # set events
        self.ui.load_trajec_btn.clicked.connect(self.load_trajectory)
        self.ui.load_param_btn.clicked.connect(self.load_params)
        self.ui.load_obj_btn.clicked.connect(self.load_obj)
        self.ui.import_btn.clicked.connect(self.setup_rendering)
        
        self.ui.start_btn.clicked.connect(self.on_start_clicked)
        self.ui.stop_btn.clicked.connect(self.on_stop_clicked)
        
        self.ui.t_slider.setMinimum(0)
        self.ui.t_slider.setMaximum(1)
        self.ui.t_slider.setSingleStep(1)
        self.ui.t_slider.valueChanged[int].connect(self.on_slider_changed)

        self.trajec_plot_model = visuals.LinePlot()

        view = self.canvas.central_widget.add_view()
        view.add(GridLines())
        view.add(XYZAxis())
        view.bgcolor = 'gray'
        view.camera = self.camera
        view.padding = 12
        self.view = view
        self.canvas.show()
        self.ui.show()
    
    def load_obj(self):
        filename = self.ui.show_file_dialog('ロケット3Dファイルを選択', './', '*.obj')
        if filename == '':
            self.ui.load_obj_title.setText('ロケット3Dモデル(obj)')
            return
        self.ui.load_obj_title.setText(filename)
        self.obj_file = filename
        return filename

    def load_trajectory(self):
        filename = self.ui.show_file_dialog('飛行履歴ファイルを選択', './', '*.csv')
        if filename == '':
            self.ui.load_trajec_title.setText('飛翔履歴ファイル(csv)')
            return
        
        self.ui.load_trajec_title.setText(filename)
        self.trajec_file = filename
        return filename
    
    def load_params(self):
        filename = self.ui.show_file_dialog('ロケットパラメータファイルを選択', './', '*.json')
        if filename == '':
            self.ui.load_param_title.setText('パラメータファイル(json)')
            return
        
        self.ui.load_param_title.setText(filename)
        self.param_file = filename
        return filename

    def setup_rendering(self):
        # パラメータ，飛翔履歴，3Dモデルを読み込んで描画設定を行う

        try:
            if self.obj_file != '':
                self.rocket_model = RocketMesh(self.obj_file)
            else:
                # デフォルトモデルを読み込み
                this_file_path = os.path.dirname(os.path.abspath(__file__))
                std_model_path = os.path.join(this_file_path, 'samples/std_scale.obj')
                print(' obj file:', std_model_path)
                # self.obj_file = std_model_path
                self.rocket_model = RocketMesh(std_model_path)
        except FileNotFoundError:
            print('obj file was not found.')
            self.ui.load_obj_title.setText('独自ロケットモデル (obj)')
            return

        try:
            if self.trajec_file != '':
                df = pd.read_csv(self.trajec_file)
            else:
                print('Trajectory file is not specified.')
                return
        except FileNotFoundError:
            print('Trajectory file: '+self.trajec_file+' was not found.')
            self.ui.load_trajec_title.setText('飛翔履歴ファイル(csv)')
            return
        
        try:
            if self.param_file != '':
                with open(self.param_file) as f:
                    self.param = json.load(f)
            else:
                print('Parameter file is not specified.')
                return
        except FileNotFoundError:
            print('Parameter file: '+self.param_file+' was not found.')
            self.ui.load_trajec_title.setText('パラメータファイル(json)')
            return
        
        self.trajec_df = df
        # 弾道履歴データを展開
        t = np.array(df['t'])
        self.t = t
        r_array = np.array(df.loc[:, 'x':'z'])
        v_array = np.array(df.loc[:, 'vx':'vz'])
        w_array = np.array(df.loc[:, 'wx':'wz'])
        q_array = np.array(df.loc[:, 'qx':'qw'])
        # 補間
        self.r = interp1d(t, r_array, kind='linear', axis=0)
        self.v = interp1d(t, v_array, kind='linear', axis=0)
        self.w = interp1d(t, w_array, kind='linear', axis=0)
        self.q = interp1d(t, q_array, kind='linear', axis=0)

        # CG値分モデルを移動
        self.rocket_model.set_CG_pos(np.array([self.param['CG_dry'], 0.0, 0.0]))

        # デフォルトのモデルを読み込む場合，モデルをスケーリング
        if self.obj_file == '':
            scale_vec = np.array([self.param['height'], self.param['diameter'], self.param['diameter']])
            self.rocket_model.set_scale(scale_vec)

        # ui内容アップデート
        self.ui.t_slider.setMaximum(int(t[-1]/self._slider_dt))
        self.rocket_model.move(self.r(0.0), self.q(0.0))
        self.trajec_plot_model = visuals.LinePlot(r_array, color='blue')
        self.view.add(self.rocket_model.visual)
        self.view.add(self.trajec_plot_model)

        # self._vertices_buffering()

        self._ready = True
        return


    def isReady(self):
        # return self._param_loaded and self._trajec_loaded
        return self._ready
    
    def update_at_t(self, t):
        if not self.isReady():
            return

        _v = self.v(t)
        _r = self.r(t)
        _w = self.w(t)
        _q = self.q(t)

        self.ui.t_text.setText(str(t))
        self.ui.r_text.setText(f"{_r[0]:.3f}, {_r[1]:.3f}, {_r[2]:.3f}")
        self.ui.v_text.setText(f"{_v[0]:.3f}, {_v[1]:.3f}, {_v[2]:.3f}")
        self.ui.w_text.setText(f"{_w[0]:.3f}, {_w[1]:.3f}, {_w[2]:.3f}")

        self.rocket_model.move(_r, _q)
        # self.rocket_model.set_vertices(self.vertices[int(t/self._slider_dt)])

        self.camera.center = _r

    def _playback_update(self):
        if self._playback_t >= self.t[-1]:
            self._playback_t = 0.0

        t = self._playback_t
        self.update_at_t(t)
        self.ui.t_slider.setValue(int(t/self._slider_dt))
        self._playback_t += self._slider_dt

    def on_start_clicked(self):
        if not self.isReady():
            return

        self._playback_mode = True
        self._playback_timer.timeout.connect(self._playback_update)
        self._playback_timer.start(10)
    
    def on_stop_clicked(self):
        if not self.isReady():
            return

        self._playback_timer.stop()
        self._playback_t = 0.0
        self.update_at_t(0)
        self.ui.t_slider.setValue(0)
        self._playback_mode = False
    
    def on_slider_changed(self, value):
        if not self.isReady():
            return

        t = value*self._slider_dt
        self.update_at_t(t)
    
    def _vertices_buffering(self):
        t_array = np.arange(0.0, self.t[-1], self._slider_dt)
        vertices_origin = (self.rocket_model.vertices + self.rocket_model.CG_pos).T
        # vertices_origin: (3, n_vertices), vertices: (n_time, n_vertices, 3)
        vertices = np.zeros((len(t_array), vertices_origin.shape[1], vertices_origin.shape[0]))
        print(' vertices buffering ')
        print(' origin vertices shape:', vertices_origin.shape)
        print(' total vertices shape: ', vertices.shape)

        i = 0
        lim = len(t_array)

        timer = QTimer(self.ui)
        def _calc_vertices():
            nonlocal i
            t = t_array[i]
            _q = quaternion.as_quat_array(self.q(t))
            Tdc = quaternion.as_rotation_matrix(_q)
            pos = self.r(t)

            vertices[i] = np.dot(Tdc, vertices_origin).T + pos
            i += 1

            self.ui.progress_bar.setValue(int(i/lim * 100))
            if i >= len(t_array):
                timer.stop()
                self._ready = True

        timer.timeout.connect(_calc_vertices)
        timer.start()
        self.vertices = vertices




if __name__ == '__main__':

    vispy.use('pyqt5')
    canvas = scene.SceneCanvas(keys="interactive", size=(1200, 800), show=False)
    camera = scene.TurntableCamera(up='+z')

    myui = MyUi()
    # myui.vispy_view.addWidget(canvas.native)

    handler = UIHandler(myui, canvas, camera)

    canvas.app.run()
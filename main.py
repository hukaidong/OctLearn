import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

from OctLearn.connector.reader import MongoCollection
from OctLearn.ScenarioTypes import ScenarioType1

mpl.use('Qt5agg')
plt.ion()


class ScenarioCasePainter():
    def __init__(self, case_id):
        coll = MongoCollection('learning', 'complete')
        doc = coll.Case_By_id(case_id)
        assert doc
        scenario = ScenarioType1(doc, r"C:\Users\Kaidong Hu\Desktop\5f8")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax_assist = fig.add_subplot(122)
        showAgents = [False for _ in range(scenario.num_agent)]

        self.showAgents = showAgents
        self.scenario = scenario
        self.figure = fig
        self.ax = ax
        self.ax_assist = ax_assist

        self.draw()

    def draw(self):
        self.ax.clear()
        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)

        self.scenario.PlotWorld(self.ax)

        for aid, show in enumerate(self.showAgents):
            if show:
                self.scenario.PlotAgentTask(self.ax, aid)
                self.scenario.PlotAgentTrajectory(self.ax, aid)

        spg = 0.3
        image = np.zeros((np.array([40, 40])/spg).astype(int))
        for aid, show in enumerate(self.showAgents):
            if show:
                self.scenario.FillByAgentTraj(image, aid, size_per_grid=spg)

        self.ax_assist.imshow(image, cmap="gray_r", origin='lower', extent=(-20, 20, -20, 20))
        self.figure.show()

    def toggleAgentId(self, aid, enabled):
        self.showAgents[aid] = enabled
        self.draw()

    def toggleAll(self, enabled):
        self.showAgents = [enabled for _ in range(self.scenario.num_agent)]
        self.draw()


import tkinter as tk
import tkinter.filedialog as fdlg


class Toggler():
    def __init__(self, root, prefix, target):
        self.enabled = True
        self.prefix = prefix
        self.target = target
        self.btn = tk.Button(root, command=self.Toggle)
        self.Toggle(dummy=True)
        self.btn.pack(side='left', expand=1, fill=tk.BOTH)

    def Toggle(self, dummy=False):
        if self.enabled:
            self.btn.configure(text=self.prefix + ' Off')
            self.enabled = False
        else:
            self.btn.configure(text=self.prefix + ' On')
            self.enabled = True
        if not dummy:
            self.target(self.enabled)

    def Force(self, enabled, dummy=True):
        self.enabled = enabled
        if self.enabled:
            self.btn.configure(text=self.prefix + ' On')
        else:
            self.btn.configure(text=self.prefix + ' Off')
        if not dummy:
            self.target(self.enabled)


def toggleWrapper(i):
    aid = i

    def func(x):
        scenario.toggleAgentId(aid, x)

    return func


if __name__ == '__main__':
    scenario = ScenarioCasePainter('5f85acee767dae76c6c9bf14')
    app = tk.Tk()
    toggles = []


    def checkFigOpen():
        if not plt.get_fignums():
            app.quit()
        app.after(100, checkFigOpen)


    for i in range(scenario.scenario.num_agent):
        if i % 10 == 0:
            frame = tk.Frame()
            frame.pack(fill='x')
        t = Toggler(frame, "Agent %d:" % i, toggleWrapper(i))
        toggles.append(t)

    def toggleAll(b):
        for t in toggles:
            t.Force(b)
        scenario.toggleAll(b)


    frame = tk.Frame()
    frame.pack()
    tk.Button(frame, text="Enable All", command=lambda: toggleAll(True)).pack(side='left')
    tk.Button(frame, text="Disable All", command=lambda: toggleAll(False)).pack(side='left')

    app.resizable(0, 0)
    app.after(500, checkFigOpen)
    tk.mainloop()

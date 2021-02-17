import numpy as np
import matplotlib as mpl
import tkinter as tk

from matplotlib import pyplot as plt
from octLearn.connector.mongo_instance import MongoInstance
from bin.scenariomanage import ScenarioType2
if __name__ == '__main__':
    mpl.use('Qt5agg')
    plt.ion()


class ScenarioCasePainter:
    def __init__(self, case_id):
        coll = MongoInstance('learning', 'complete')
        doc = coll.Case_By_id(case_id)
        assert doc
        sc = ScenarioType2(doc)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        ax_assist = fig.add_subplot(122)
        showAgents = [False for _ in range(sc.num_agent)]

        self.showAgents = showAgents
        self.scenario = sc
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
        image = np.zeros((np.array([40, 40]) / spg).astype(int))
        for aid, show in enumerate(self.showAgents):
            if show:
                print(self.scenario.agent_trajectories[aid].shape)
                self.scenario.FillByAgentTraj(image, aid, size_per_grid=spg)

        self.ax_assist.imshow(image.T, cmap="gray_r", origin='upper', extent=(-20, 20, -20, 20))
        self.figure.show()

    def toggleAgentId(self, aid, enabled):
        self.showAgents[aid] = enabled
        self.draw()

    def toggleAll(self, enabled):
        self.showAgents = [enabled for _ in range(self.scenario.num_agent)]
        self.draw()


class Toggler:
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


def toggleWrapper(aid):
    _aid = aid

    def func(x):
        scenario.toggleAgentId(_aid, x)

    return func


if __name__ == '__main__':
    scenario = ScenarioCasePainter('5f85adca767dae76c6c9bf17')
    app = tk.Tk()
    toggles = []


    def checkFigOpen():
        if not plt.get_fignums():
            app.quit()
        app.after(100, checkFigOpen)


    frame = None
    for i in range(scenario.scenario.num_agent):
        if i % 10 == 0 or frame is None:
            frame = tk.Frame()
            frame.pack(fill='x')
        toggler = Toggler(frame, "Agent %d:" % i, toggleWrapper(i))
        toggles.append(toggler)


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

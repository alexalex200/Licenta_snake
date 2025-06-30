import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'

class Plot:
    def __init__(self):
        self.score_ppo = []
        self.steps_ppo = []

        self.score_ga = []
        self.steps_ga = []

        self.score_naiv = []
        self.steps_naiv = []

    def add_ppo(self, score, steps):
        self.score_ppo.append(score)
        self.steps_ppo.append(steps)

    def add_ga(self, score, steps):
        self.score_ga.append(score)
        self.steps_ga.append(steps)

    def add_naiv(self, score, steps):
        self.score_naiv.append(score)
        self.steps_naiv.append(steps)

    def save_data(self):
        if len(self.score_ppo) > 0 and len(self.steps_ppo) > 0:
            with open('data/ppo_data.txt', 'w') as f:
                for score, steps in zip(self.score_ppo, self.steps_ppo):
                    f.write(f"{score},{steps}\n")
        if len(self.score_ga) > 0 and len(self.steps_ga) > 0:
            with open('data/ga_data.txt', 'w') as f:
                for score, steps in zip(self.score_ga, self.steps_ga):
                    f.write(f"{score},{steps}\n")
        if len(self.score_naiv) > 0 and len(self.steps_naiv) > 0:
            with open('data/naiv_data.txt', 'w') as f:
                for score, steps in zip(self.score_naiv, self.steps_naiv):
                    f.write(f"{score},{steps}\n")

    def load_data(self):
        if os.path.exists('data/ppo_data.txt'):
            with open('data/ppo_data.txt', 'r') as f:
                for line in f:
                    score, steps = map(int, line.strip().split(','))
                    self.add_ppo(score, steps)
        if os.path.exists('data/ga_data.txt'):
            with open('data/ga_data.txt', 'r') as f:
                for line in f:
                    score, steps = map(int, line.strip().split(','))
                    self.add_ga(score, steps)
        if os.path.exists('data/naiv_data.txt'):
            with open('data/naiv_data.txt', 'r') as f:
                for line in f:
                    score, steps = map(int, line.strip().split(','))
                    self.add_naiv(score, steps)

    def plot_training(self):
        self.load_data()
        list_for_plot = []
        for i in range(0, len(self.score_ppo), 20):
            list_for_plot.append(sum(self.score_ppo[i:i + 20])//20)

        # fig = px.line(
        #     x=list(range(len(list_for_plot))),
        #     y=list_for_plot,
        #     labels={'x': 'Jocuri', 'y': 'Scor'},
        #     title='Grafic antrenare PPO'
        # )

        fig2 = px.line(
            x=list(range(len(self.score_ga))),
            y=self.score_ga,
            labels={'x': 'Generatii', 'y': 'Scor'},
            title='Grafic antrenare GA'
        )

        fig2.update_layout(
            title_font_size=60,  # Titlu
            font=dict(size=40),  # Text general
            xaxis=dict(title_font_size=40, tickfont_size=40),  # Axe X
            yaxis=dict(title_font_size=40 , tickfont_size=40)  # Axe Y
        )

        fig2.show()

    def plot_testing(self):
        self.load_data()

        plot_ppo = []
        for i in range(0, len(self.score_ppo)-5):
            plot_ppo.append(sum(self.score_ppo[i:i + 5]) // 5)
        for i in range(5):
            plot_ppo.append(plot_ppo[-1])

        plot_ga = []
        for i in range(0, len(self.score_ga)-5):
            plot_ga.append(sum(self.score_ga[i:i + 5]) // 5)
        for i in range(5):
            plot_ga.append(plot_ga[-1])

        plot_naiv = [33] * len(plot_ppo)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(plot_ppo))),
            y=plot_ppo,
            mode='lines+markers',
            name='PPO',
            line=dict(width=5)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(plot_ga))),
            y=plot_ga,
            mode='lines+markers',
            name='GA',
            line=dict(width=5)
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(plot_naiv))),
            y=plot_naiv,
            mode='lines+markers',
            name='Naiv',
            line=dict(width=5)
        ))
        fig.update_layout(
            title='Grafic testare',
            xaxis_title='Jocuri',
            yaxis_title='Scor',
            legend=dict(title='Algoritmi'),
            width=1500,
            height=800
        )

        fig.update_layout(
            title_font_size=60,  # Titlu
            font=dict(size=40),  # Text general
            xaxis=dict(title_font_size=40, tickfont_size=40),  # Axe X
            yaxis=dict(title_font_size=40, tickfont_size=40)  # Axe Y
        )
        fig.show()

    def plot_columns(self):
        #ga7815
        #ppo1857
        self.load_data()

        fig = px.bar(
            x=['PPO', 'GA', 'Naiv'],
            y=[sum(self.steps_ppo)/1000, sum(self.steps_ga)/1000, sum(self.score_naiv)/1000],
            labels={'x': 'Algoritmi', 'y': 'Numar pasi'},
            title='Numar pasi mediu pentru finalizarea cu success a jocului',
        )

        fig.update_traces(marker_color=['blue', 'green', 'red'], marker_line_color='black', marker_line_width=2)
        fig.update_layout(
            title_font_size=40,  # Titlu
            font=dict(size=40),  # Text general
            xaxis=dict(title_font_size=40, tickfont_size=40),  # Axe X
            yaxis=dict(title_font_size=40, tickfont_size=40)  # Axe Y
        )

        fig.show()


if __name__ == "__main__":
    plot = Plot()
    plot.plot_training()


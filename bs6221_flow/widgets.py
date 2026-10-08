"""Notebook controls call the same Python API as scripted experiments."""
import ipywidgets as w
from IPython.display import display,clear_output
from .sampling import generate_images,compare_solvers
from .presentation import noise_preview
from .numerics import compare_numerics


def generation_panel(net):
    digit=w.Dropdown(options=list(range(10)),value=3,description='数字')
    seed=w.BoundedIntText(value=42,min=0,max=2**31-1,description='噪声种子')
    scale=w.FloatSlider(value=1.,min=.2,max=2.,step=.1,description='噪声尺度',continuous_update=False)
    method=w.Dropdown(options=['Euler','Heun','RK4'],description='求解器')
    steps=w.BoundedIntText(value=40,min=1,max=2048,description='步数')
    count=w.BoundedIntText(value=4,min=1,max=32,description='图片数')
    budget=w.BoundedIntText(value=40,min=4,max=2048,step=4,description='等 NFE')
    preview=w.Button(description='预览所选噪声')
    generate=w.Button(description='生成并展示轨迹',button_style='primary')
    compare=w.Button(description='相同噪声比较求解器')
    output=w.Output();state={}
    def invoke(action):
        preview.disabled=generate.disabled=compare.disabled=True
        try:
            with output:
                clear_output(wait=True)
                noise_preview(seed.value,scale.value)
                if action=='generate':
                    state['result']=generate_images(digit.value,count.value,method.value,steps.value,seed.value,scale.value,trajectory=True,net=net)
                    print(state['result']['params'])
                if action=='compare':
                    state['comparison']=compare_solvers(digit.value,count.value,nfe_budget=budget.value,seed=seed.value,noise_scale=scale.value,net=net)
                    print('Same labels and noise; image differences are not a numerical error estimate.')
        except (ValueError,RuntimeError) as error:
            with output:print(str(error))
        finally:preview.disabled=generate.disabled=compare.disabled=False
    preview.on_click(lambda _:invoke('preview'));generate.on_click(lambda _:invoke('generate'));compare.on_click(lambda _:invoke('compare'))
    panel=w.VBox([w.HTML('<b>选择数字与初始噪声</b>：种子固定噪声，默认尺度 1 为标准高斯。'),
                  w.HBox([digit,seed]),w.HBox([scale,count]),w.HBox([method,steps,budget]),w.HBox([preview,generate,compare]),output])
    panel.flow_state=state
    panel.flow_controls={'digit':digit,'seed':seed,'scale':scale,'count':count,'method':method,'steps':steps,'budget':budget,'generate':generate,'compare':compare,'preview':preview}
    return panel


def numerics_panel(net):
    digit=w.Dropdown(options=list(range(10)),value=3,description='数字')
    seed=w.BoundedIntText(value=42,min=0,max=2**31-1,description='种子')
    count=w.BoundedIntText(value=2,min=1,max=100,description='样本数')
    steps=w.Text(value='4,8,16,32',description='步数列表')
    budgets=w.Text(value='4,8,20,40',description='NFE 列表')
    refs=w.Text(value='128,256',description='参考步数')
    repeats=w.BoundedIntText(value=3,min=1,max=10,description='计时重复')
    run=w.Button(description='重新计算小规模实验',button_style='warning');output=w.Output();state={}
    def calculate(_):
        run.disabled=True
        try:
            with output:
                clear_output(wait=True)
                parse=lambda field:tuple(int(x.strip()) for x in field.value.split(','))
                state['result']=compare_numerics(net,digit.value,seed.value,count.value,parse(steps),parse(budgets),parse(refs),repeats.value)
        except (ValueError,RuntimeError) as error:
            with output:print(str(error))
        finally:run.disabled=False
    run.on_click(calculate)
    panel=w.VBox([w.HTML('<b>ODE 数值分析</b>：固定模型、噪声和标签。参考步数越大越耗时。'),
                  w.HBox([digit,seed,count]),w.HBox([steps,budgets]),w.HBox([refs,repeats]),run,output])
    panel.flow_state=state;panel.flow_controls={'run':run,'steps':steps,'budgets':budgets,'refs':refs,'count':count,'seed':seed,'digit':digit,'repeats':repeats}
    return panel

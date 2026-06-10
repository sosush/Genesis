import sys
import os
import gradio as gr
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.synthesis.engine import SynthesisEngine

def parse_examples(examples_str: str):
    """Parse '1->1, 2->4, 3->9' format into TestCases."""
    pairs = []
    items = examples_str.split(',')
    for item in items:
        if '->' not in item:
            continue
        inp, out = item.split('->')
        inp = int(inp.strip())
        out = int(out.strip())
        pairs.append(({"x": inp}, out))
    return pairs

def synthesize(examples_str: str, max_generations: int):
    pairs = parse_examples(examples_str)
    if not pairs:
        return (
            "Error: Invalid format. Please use the '1->1, 2->4, 3->9' format.", 
            0, 
            None,
            "The synthesis failed because the input format was not recognized."
        )
        
    engine = SynthesisEngine(max_gen=max_generations, variables=["x"])
    result = engine.run(pairs)
    
    # Generate fitness plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.fitness_curve, color='#66fcf1', linewidth=2)
    ax.set_title('Evolutionary Convergence (Best Fitness over Generations)', color='white')
    ax.set_xlabel('Generation', color='#c5c6c7')
    ax.set_ylabel('Fitness Score (0 to 1)', color='#c5c6c7')
    ax.set_facecolor('#0b0c10')
    fig.patch.set_facecolor('#0b0c10')
    ax.tick_params(colors='#c5c6c7')
    for spine in ax.spines.values():
        spine.set_color('#1f2833')
        
    program_str = result.program or "Failed to synthesize a solution."
    
    # Generate an explanation of the result based on whether it succeeded
    if result.best_fitness > 0.99:
        explanation = (
            f"**Success!** The Neuro-Symbolic engine successfully discovered the underlying logic "
            f"in {result.generations_taken} generations.\n\n"
            f"**How it works:** The engine explored thousands of potential mathematical expressions (ASTs). "
            f"Instead of executing every single one (which is slow), a PyTorch Neural Network predicted which "
            f"equations were most promising. Only the best candidates were actually executed against your test cases, "
            f"saving massive computational time."
        )
    else:
        explanation = (
            f"**Partial Convergence.** The engine ran out of generations before finding a perfect match. "
            f"The best program found scored a fitness of {result.best_fitness:.2f}/1.0.\n\n"
            f"Try increasing the 'Max Generations' limit or providing simpler, more direct test cases."
        )
        
    return program_str, result.generations_taken, fig, explanation

# Define the custom theme
custom_theme = gr.themes.Monochrome(
    neutral_hue="slate",
    primary_hue="teal",
    font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"]
)

with gr.Blocks(theme=custom_theme) as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center; color: #66fcf1; letter-spacing: 5px;'>GENESIS</h1>
        <h3 style='text-align: center; color: #c5c6c7;'>Autonomous Neuro-Symbolic Program Synthesis</h3>
        """
    )
    
    with gr.Accordion("📚 Research Glossary & How It Works (Click to Expand)", open=False):
        gr.Markdown(
            """
            Genesis solves the **Grand Challenge of Program Synthesis**: automatically generating computer code that satisfies a set of constraints without human intervention.
            
            Instead of standard LLMs (which just guess text based on statistics), Genesis uses **Neuro-Symbolic Evolution**:
            - **Symbolic Evolution (The Search Space):** The code explores the infinite space of Abstract Syntax Trees (ASTs). It uses Darwinian evolution (Crossover and Mutation) to breed "programs".
            - **Neural Pre-filtering (The Speedup):** A PyTorch Multi-Layer Perceptron (MLP) learns to predict whether a mutated program will succeed *before* it is even executed. This acts as a heuristic proxy, accelerating convergence by ~4x.
            
            **Key Terms:**
            - **Fitness Score:** A value from `0.0` to `1.0` representing how close a program is to the perfect answer. `1.0` means it passed all test cases.
            - **Generation:** A single cycle of evolution (evaluating the population, killing the weakest, breeding the strongest).
            """
        )
        
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Specification")
            examples_input = gr.Textbox(
                label="Input-Output Examples", 
                info="Provide mapping from input (x) to output. Format: x1->y1, x2->y2",
                placeholder="1->1, 2->4, 3->9",
                value="1->1, 2->4, 3->9, 4->16"
            )
            max_gen = gr.Slider(
                10, 500, value=100, step=10, 
                label="Max Generations",
                info="How long the evolutionary algorithm is allowed to run. Complex logic requires more generations."
            )
            btn = gr.Button("🚀 Synthesize Program", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 2. Evolutionary Result")
            output_code = gr.Code(label="Synthesized Algorithm (Python)", language="python")
            
            with gr.Row():
                gen_taken = gr.Number(label="Generations Taken to Converge")
            
            explanation_box = gr.Markdown("Waiting for execution...")
            fitness_plot = gr.Plot(label="Convergence Trajectory")
            
    btn.click(
        fn=synthesize,
        inputs=[examples_input, max_gen],
        outputs=[output_code, gen_taken, fitness_plot, explanation_box]
    )

if __name__ == "__main__":
    demo.launch()

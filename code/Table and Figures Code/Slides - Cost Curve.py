import numpy as np
import matplotlib.pyplot as plt

def human_cost(num_municipalities, num_questions):
    return 2.50 * num_municipalities * num_questions

def llm_api_cost(num_municipalities, num_questions):
    return 0.03 * num_municipalities * num_questions

def llm_question_prep_cost(num_questions):
    return 390 * num_questions


def llm_text_prep_cost(num_municipalities):
    average_cost = (4800 + 8400 + 12000) / 3
    return average_cost / num_municipalities * num_municipalities




def llm_total_cost(num_municipalities, num_questions):
    return (llm_api_cost(num_municipalities, num_questions) +
            llm_question_prep_cost(num_questions) +
            llm_text_prep_cost(num_municipalities))

# Create the main figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 6))

# Line chart
municipalities = np.linspace(0, 3000, 200)
questions_low = 15
questions_high = 50

human_costs_low = human_cost(municipalities, questions_low)/1000
human_costs_high = human_cost(municipalities, questions_high)/1000
llm_costs_low = llm_total_cost(municipalities, questions_low)/1000
llm_costs_high = llm_total_cost(municipalities, questions_high)/1000

ax1.plot(municipalities, human_costs_low, label=f'Human ({questions_low} questions)', color='blue')
ax1.plot(municipalities, human_costs_high, label=f'Human ({questions_high} questions)', color='lightblue')
ax1.plot(municipalities, llm_costs_low, label=f'LLM ({questions_low} questions)', color='red')
ax1.plot(municipalities, llm_costs_high, label=f'LLM ({questions_high} questions)', color='lightcoral')

ax1.set_xlabel('Number of Municipalities')
ax1.set_ylabel('Total Cost (1,000$)')
ax1.set_title('Cost Comparison: Human vs. LLM Process')
ax1.legend()

# Bar chart
municipalities_sample = 3000
questions_sample = [15, 50]
bar_width = 0.35
index = np.arange(len(questions_sample))

llm_api_costs = [llm_api_cost(municipalities_sample, q)/1000 for q in questions_sample]
llm_question_prep_costs = [llm_question_prep_cost(q)/1000 for q in questions_sample]
llm_text_prep_costs = llm_text_prep_cost(municipalities_sample)/1000

llm_bottom = np.zeros(len(questions_sample))
ax2.bar(index, llm_api_costs, bar_width, label='LLM Cost to Call API', color='red', bottom=llm_bottom)
llm_bottom += llm_api_costs
ax2.bar(index, llm_question_prep_costs, bar_width, label='Human Cost to Prepare Questions', color='lightcoral', bottom=llm_bottom)
llm_bottom += llm_question_prep_costs
ax2.bar(index, [llm_text_prep_costs] * len(questions_sample), bar_width, label='Human Cost to Prepare Text', color='pink', bottom=llm_bottom)

ax2.set_xlabel('Number of Questions')
ax2.set_ylabel('Total Cost (1,000$)')
ax2.set_title(f'LLM Cost Breakdown\n({municipalities_sample} Municipalities)')
ax2.set_xticks(index)
ax2.set_xticklabels(questions_sample)
ax2.legend()

plt.tight_layout()

#Save figure in r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\results\figures"
plt.savefig(r"C:\Users\Dan's Laptop\Dropbox\Inclusionary Zoning\Github\ai-zoning\results\figures\Slides - Cost Curve.png", dpi = 300)

plt.show()
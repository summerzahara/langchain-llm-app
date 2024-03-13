# from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
# from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import load_tools, initialize_agent, AgentType
from dotenv import load_dotenv
from icecream import ic

load_dotenv()


def gen_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)

    # name = llm("I have a cat per and I want a cool name for it. suggest me five cool names for my pet.")
    # return name

    prompt_template = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template=f'I have a {animal_type} pet and I want a cool name for it. It is {pet_color} in color. Suggest me '
                 f'five cool names for '
                 f'my '
                 f'pet.'
    )

    name_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        output_key='pet_name'  # This is what make the output readable
    )

    response = name_chain(
        {
            'animal_type': animal_type,
            'pet_color': pet_color,
        }
    )
    return response


def langchain_agent():
    llm = OpenAI(temperature=0.5)

    tools = load_tools(
        ['wikipedia', 'llm-math'],
        llm=llm
    )

    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # result = agent.run(
    #     'What is the average age of a cat? Multiply the age by 3.'
    # )

    result = agent.invoke(
        {'input': 'What is the average age of a cat? Multiply the age by 3.'}
    )
    return result


if __name__ == '__main__':
    # ic(gen_pet_name('cat', 'orange'))
    ic(langchain_agent())

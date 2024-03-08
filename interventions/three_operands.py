from .intervention import Intervention
import json
import random
from tqdm import tqdm
from utils.number_utils import is_int, convert_to_words
from interventions.entity_intervention import get_entity_data
from interventions.lama_intervention import get_lama_data
from transformers import AutoTokenizer
import sys

INTERVENTION_TYPES_TWO_RESULTS = [1, 3, 10, 11]
INTERVENTION_TYPES_SINGLE_RESULT = [2, 20]
INTERVENTION_TYPES = INTERVENTION_TYPES_TWO_RESULTS + INTERVENTION_TYPES_SINGLE_RESULT


def generate_operands_triple(args, equation, keep_result):
    llama_setting = False
    if args.max_n == 9:
        llama_setting = True
        op_max = 50
        res_max = 9
    else:
        op_max = args.max_n
        res_max = args.max_n

    num_retries = 0
    while 1:
        x_base = str(random.randint(2, op_max))
        y_base = str(random.randint(2, op_max))
        z_base = str(random.randint(2, op_max))
        try:
            res_base = eval(equation.replace('{x}', x_base).replace('{y}', y_base).replace('{z}', z_base))
        except ZeroDivisionError:
            continue
        if not is_int(res_base) or not res_base in range(1, res_max):
            continue

        res_base = str(int(res_base))
        res_alt = None

        while 1:
            x_alt = str(random.randint(1, op_max))
            y_alt = str(random.randint(1, op_max))
            z_alt = str(random.randint(1, op_max))
            num_retries += 1

            try:
                res_alt = eval(equation.replace('{x}', x_alt).replace('{y}', y_alt).replace('{z}', z_alt))
            except ZeroDivisionError:
                continue
            if is_int(res_alt) and res_alt in range(1, res_max):
                if llama_setting:
                    if len(str(x_base)) != len(str(x_alt)) or len(str(y_base)) != len(str(y_alt)) or len(
                            str(z_base)) != len(str(z_alt)):
                        continue
                    if not keep_result and int(res_alt) == int(res_base):
                        continue

                if not keep_result:
                    break
                elif int(res_alt) == int(res_base):
                    break

                if num_retries % 10000 == 0:
                    break
                if num_retries > 1000000000:
                    raise Exception(
                        f'Could not find a pair of operands with the same result:\
                        {equation.replace("{x}", x_base).replace("{y}", y_base).replace("{z}", z_base)} = {res_base}')

        if res_alt is not None:
            break

    res_alt = str(int(res_alt))

    return [(x_base, y_base, z_base, res_base), (x_alt, y_alt, z_alt, res_alt)]

def get_arithmetic_data_three_operands(tokenizer, args):
    if not args.extended_templates:
        with open('interventions/new_three_operand_questions.json') as fp:
            three_operand_questions = json.load(fp)
        with open('interventions/few_shot_three_operand_questions.json') as fsp:
            few_shot_templates = json.load(fsp)
    elif args.extended_templates:
        if not args.reversed_fewshot:
            with open('interventions/new_three_operand_questions_extended.json') as fp:
                three_operand_questions = json.load(fp)
            with open('interventions/few_shot_three_operand_questions_extended.json') as fsp:
                few_shot_templates = json.load(fsp)
        elif args.reversed_fewshot:
            with open('interventions/new_three_operand_questions_extended.json') as fp:
                three_operand_questions = json.load(fp)
            with open('interventions/few_shot_three_operand_questions_extended_reversed.json') as fsp:
                few_shot_templates = json.load(fsp)

    keep_result = args.intervention_type in INTERVENTION_TYPES_SINGLE_RESULT
    keep_result = False
    is_opt = args.model.startswith('facebook/opt')
    is_bloom = args.model.startswith('bigscience/bloom')
    is_mistral = args.model.startswith('mistralai')
    is_persimmon = args.model.startswith('persimmon')

    intervention_list = []
    progress = tqdm(total=len(three_operand_questions) * args.examples_per_template)

    # for equation, template in three_operand_questions.items():
    #     few_shots = ''
        # if args.n_shots > 0:
        #     fs_template = few_shot_templates.get(equation)
        #     for _ in range(args.n_shots):
        #         base_tuple, alt_tuple = generate_operands_triple(args, equation, keep_result)
        #         print('base_tuple: ', base_tuple)
        #         x_base, y_base, z_base, res_base = base_tuple
        #         shot = fs_template.replace('{x}', x_base).replace('{y}', y_base).replace('{z}', z_base) + f'{res_base}.'
        #         few_shots += shot
    for equation, template in three_operand_questions.items():
        for _ in range(args.examples_per_template):
            few_shots = ''
            if args.n_shots > 0:
                fs_template = few_shot_templates.get(equation)
                for _ in range(args.n_shots):
                    fs_base_tuple, _ = generate_operands_triple(args, equation, keep_result)
                    fs_x_base, fs_y_base, fs_z_base, fs_res_base = fs_base_tuple
                    if args.representation == 'words':
                        fs_x_base = convert_to_words(fs_x_base)
                        fs_y_base = convert_to_words(fs_y_base)
                        fs_z_base = convert_to_words(fs_z_base)
                        fs_res_base = convert_to_words(fs_res_base)
                    if not args.reversed_fewshot:
                        shot = fs_template.replace('{x}', fs_x_base).replace('{y}', fs_y_base).replace('{z}', fs_z_base) + f'{fs_res_base}. '
                    elif args.reversed_fewshot:
                        shot = f'{fs_res_base}' + fs_template.replace('{x}', fs_x_base).replace('{y}', fs_y_base).replace('{z}', fs_z_base)
                    few_shots += shot

            base_tuple, alt_tuple = generate_operands_triple(args, equation, keep_result)
            x_base, y_base, z_base, res_base = base_tuple
            x_alt, y_alt, z_alt, res_alt = base_tuple # alt_tuple
            if args.representation == 'words':
                x_base = convert_to_words(x_base)
                y_base = convert_to_words(y_base)
                z_base = convert_to_words(z_base)
                x_alt = convert_to_words(x_alt)
                y_alt = convert_to_words(y_alt)
                z_alt = convert_to_words(z_alt)
                res_base = convert_to_words(res_base)
                res_alt = convert_to_words(res_alt)
            base_string = template.replace('{x}', x_base).replace('{y}', y_base).replace('{z}', z_base)
            alt_string = template.replace('{x}', x_alt).replace('{y}', y_alt).replace('{z}', z_alt)
            
            
            intervention = Intervention(tokenizer,
                                        template_type='-',
                                        base_string=base_string,
                                        alt_string=alt_string,
                                        equation=equation,
                                        few_shots=few_shots,
                                        n_vars=2,
                                        is_opt=is_opt,
                                        is_bloom=is_bloom,
                                        is_mistral=is_mistral,
                                        is_persimmon=is_persimmon,
                                        representation=args.representation,
                                        extended_templates=args.extended_templates)
            intervention.set_results(res_base, res_alt)
            intervention.set_position_of_tokens_three_operands((x_base, y_base, z_base), (x_alt, y_alt, z_alt))
            intervention_list.append(intervention)
            progress.update()

    return intervention_list
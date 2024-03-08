import torch
import pdb
import sys

class Intervention:
    """
    Wrapper for all the possible interventions
    """

    def __init__(self,
                 tokenizer,
                 template_type,
                 base_string: str,
                 alt_string: str,
                 equation: str,
                 n_vars,
                 is_opt,
                 is_bloom,
                 is_mistral,
                 is_persimmon,
                 representation,
                 extended_templates,
                 few_shots='',
                 few_shots_t2=None,
                 multitoken=False,
                 device='cpu'
                 ):
        self.op3_pos = None
        self.operator_word = None
        self.operands_alt = None
        self.operands_base = None
        self.operator_pos = None
        self.op2_pos = None
        self.op1_pos = None
        self.res_alt_tok = None
        self.res_base_tok = None
        self.res_string = None
        self.res_base_string = None
        self.res_alt_string = None
        self.device = device
        self.multitoken = multitoken
        self.is_llama = False
        self.is_opt = is_opt
        self.is_bloom = is_bloom
        self.is_mistral = is_mistral
        self.is_persimmon = is_persimmon
        self.representation = representation
        self.extended_templates = extended_templates

        self.template_id = template_type
        self.n_vars = n_vars

        # All the initial strings
        self.base_string = base_string
        self.alt_string = alt_string
        self.few_shots = few_shots
        if few_shots_t2 is not None:
            self.few_shots_t2 = few_shots_t2
        else:
            self.few_shots_t2 = ' ' # few_shots: need to add space to make the tokenization equal to base prompt.

        self.equation = equation

        self.enc = tokenizer

        if self.enc is not None:
            self.is_llama = ('llama' in self.enc.name_or_path or 'alpaca' in self.enc.name_or_path)
            add_sp_tokens = True if 'google/flan' in self.enc.name_or_path else False
            if self.is_bloom or self.is_mistral or self.is_persimmon:
                add_sp_tokens = True
            if self.is_llama:
                base_string += ' '
                alt_string += ' '
            self.len_few_shots = len(self.enc.encode(self.few_shots))
            self.len_few_shots_t2 = len(self.enc.encode(self.few_shots_t2))
            # encode few shot examples and task tokens
            if self.len_few_shots == 0:
                if self.is_bloom or self.is_mistral:
                    self.base_string_tok_list = self.enc.encode(tokenizer.bos_token + base_string, add_special_tokens=add_sp_tokens)
                    self.alt_string_tok_list = self.enc.encode(tokenizer.bos_token + alt_string, add_special_tokens=add_sp_tokens)
                else:
                    self.base_string_tok_list = self.enc.encode(' ' + base_string, add_special_tokens=add_sp_tokens)
                    self.alt_string_tok_list = self.enc.encode(' ' + alt_string, add_special_tokens=add_sp_tokens)
            else:
                if self.is_bloom or self.is_mistral:
                    self.base_string_tok_list = self.enc.encode(tokenizer.bos_token + self.few_shots + base_string, add_special_tokens=add_sp_tokens)
                    self.alt_string_tok_list = self.enc.encode(self.few_shots_t2 + alt_string, add_special_tokens=add_sp_tokens)
                else:
                    self.base_string_tok_list = self.enc.encode(self.few_shots + base_string, 
                                                                add_special_tokens=add_sp_tokens)
                    self.alt_string_tok_list = self.enc.encode(self.few_shots_t2 + alt_string, add_special_tokens=add_sp_tokens)
            #print("base string tok list: ", self.base_string_tok_list)
            #print("alt string tok list: ", self.alt_string_tok_list)
            max_length = max(len(self.base_string_tok_list), len(self.alt_string_tok_list))
            padding_length = max_length - len(self.alt_string_tok_list)
            if self.is_opt:
                if padding_length == 0:
                    pass
                else:
                    padded_tokens = [self.enc.pad_token_id] * (padding_length - 1)
                    padded_tokens += [self.enc.bos_token_id]
                    padded_tokens += self.alt_string_tok_list
                    self.alt_string_tok_list = padded_tokens
            elif self.is_bloom:
                if padding_length == 1:
                    pass
                else:
                    padded_tokens = [self.enc.bos_token_id]
                    padded_tokens += [self.enc.pad_token_id] * (padding_length - 2)
                    padded_tokens += [self.enc.bos_token_id]
                    padded_tokens += self.alt_string_tok_list
                    self.alt_string_tok_list = padded_tokens
            elif self.is_mistral:
                if padding_length == 1:
                    pass
                else:
                    padded_tokens = [self.enc.bos_token_id]
                    padded_tokens += [self.enc.encode(' ')[0]] * (padding_length - 2)
                    padded_tokens += [self.enc.bos_token_id]
                    padded_tokens += self.alt_string_tok_list
                    self.alt_string_tok_list = padded_tokens
            else:
                if padding_length == 0:
                    pass
                else:
                    #padded_tokens = self.enc.encode("Solve this arithmetic task by outputting a number.")
                    padded_tokens = [self.enc.encode(' ')[0]] * (padding_length - 1) # design choice to prepend whitespaces
                    padded_tokens += [self.enc.bos_token_id] #* padding_length
                    padded_tokens += self.alt_string_tok_list
                    self.alt_string_tok_list = padded_tokens

            self.base_string_tok = torch.LongTensor(self.base_string_tok_list).to(device).unsqueeze(0)
            self.alt_string_tok = torch.LongTensor(self.alt_string_tok_list).to(device).unsqueeze(0)

            assert len(self.base_string_tok) == len(self.alt_string_tok), '{} vs {}'.format(
                self.base_string, self.alt_string)

    def set_results(self, res_base, res_alt):
        self.res_base_string = res_base
        self.res_alt_string = res_alt

        if self.enc is not None:
            if 'google/flan' in self.enc.name_or_path:
                self.res_base_tok = ['▁' + res_base]
                self.res_alt_tok = ['▁' + res_alt]
                if not self.multitoken:
                    assert len(self.enc.convert_tokens_to_ids(self.res_base_tok)) == 1, '{} - {}'.format(
                        self.enc.tokenize(self.res_base_tok), res_base)
                    assert len(self.enc.convert_tokens_to_ids(self.res_alt_tok)) == 1, '{} - {}'.format(
                        self.enc.tokenize(self.res_alt_tok), res_alt)

            else:
                # 'a ' added to input so that tokenizer understands that first word
                # follows a space.
                if self.is_llama:
                    prefix = ''
                elif self.is_opt:
                    prefix = 'a '
                    start_idx = 1
                    if self.representation == "arabic":
                        self.res_base_tok = self.enc.tokenize(prefix + res_base)[start_idx:]
                        self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[start_idx:]
                    else:
                        self.res_base_tok = self.enc.tokenize(prefix + res_base)[start_idx:]
                        self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[start_idx:]
                elif self.is_bloom or self.is_mistral:
                    prefix = 'a '
                    start_idx = 1
                    if self.representation == "arabic":
                        self.res_base_tok = self.enc.tokenize(prefix + res_base)[start_idx:]
                        self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[start_idx:]
                    else:
                        self.res_base_tok = self.enc.tokenize(prefix + res_base)[start_idx:]
                        self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[start_idx:]
                else:
                    # if self.representation == 'words':
                    #     prefix = ' '
                    #     self.res_base_tok = self.enc.tokenize(prefix  + res_base)[1:]
                    #     self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[1:]
                    if self.representation == 'words' and self.extended_templates:
                        prefix = 'a '
                        self.res_base_tok = self.enc.tokenize(prefix  + res_base)[1:]
                        self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[1:]
                    elif self.representation == 'arabic' and not self.extended_templates:
                        prefix = ''#'a '
                        self.res_base_tok = self.enc.tokenize(prefix  + res_base)#[1:]
                        self.res_alt_tok = self.enc.tokenize(prefix + res_alt)#[1:]
                    elif self.representation == 'arabic' and self.extended_templates:
                        prefix = 'a '
                        self.res_base_tok = self.enc.tokenize(prefix  + res_base)[1:]
                        self.res_alt_tok = self.enc.tokenize(prefix + res_alt)[1:]
                if not self.multitoken:
                    if not self.is_bloom and not self.is_mistral:
                        print(f"res base: {res_base}")
                        print(f"res base wo whitespace: {res_base[1:]}")
                        print(f"res base with prefix: {prefix + res_base}")
                        print(f"res base tok: {self.enc.tokenize(prefix + res_base)}")
                        assert len(self.res_base_tok) == 1, '{} - {}'.format(self.res_base_tok, res_base)
                        assert len(self.res_alt_tok) == 1, '{} - {}'.format(self.res_alt_tok, res_alt)
                    else:
                        print(f"res base: {res_base}")
                        print(f"res base wo whitespace: {res_base[1:]}")
                        print(f"res base with prefix: {prefix + res_base}")
                        print(f"res base tok: {self.enc.tokenize(prefix + res_base)}")
                        assert len(self.enc.encode(prefix + res_base)[1:]) == 1, '{} - {}'.format(self.enc.encode(prefix + res_base), res_base)
                        assert len(self.enc.encode(prefix + res_alt)[1:]) == 1, '{} - {}'.format(self.enc.encode(prefix + res_alt), res_alt)

            self.res_base_tok = self.enc.convert_tokens_to_ids(self.res_base_tok)
            self.res_alt_tok = self.enc.convert_tokens_to_ids(self.res_alt_tok)

    def set_predicted_alt_result(self, pred_alt_string, pred_res_alt_tok):
        self.pred_alt_string = pred_alt_string
        self.pred_res_alt_tok = pred_res_alt_tok



    @staticmethod
    def index_last_occurrence(lst, item):
        return len(lst) - lst[::-1].index(item) - 1

    def set_position_of_tokens(self, operands_base, operands_alt, operator_word, no_space_before_op1=False):
        self.operands_base = ' '.join(operands_base)
        self.operands_alt = ' '.join(operands_alt)
        self.operator_word = operator_word
        x_base, y_base = operands_base
        if self.is_llama:
            # todo for llama take the last token of the number
            x_base_tok = self.enc.tokenize(x_base)[-1:]
            y_base_tok = self.enc.tokenize(y_base)[-1:]
        else:
            prefix = '\n' if no_space_before_op1 else ''#'a '
            x_base_tok = self.enc.tokenize(prefix + '' + x_base)#[1:]
            y_base_tok = self.enc.tokenize(prefix + y_base)#[1:]
        assert len(x_base_tok) == 1, '{} - {}'.format(x_base_tok, x_base)
        assert len(y_base_tok) == 1, '{} - {}'.format(y_base_tok, y_base)
        x_base_tok = self.enc.convert_tokens_to_ids(x_base_tok)[0]
        y_base_tok = self.enc.convert_tokens_to_ids(y_base_tok)[0]

        operator_word_tok = self.enc.tokenize(prefix + operator_word)#[1:]
        assert len(operator_word_tok) == 1, '{} - {}'.format(operator_word_tok, operator_word)
        operator_word_tok = self.enc.convert_tokens_to_ids(operator_word_tok)[0]
        self.op2_pos = self.index_last_occurrence(self.base_string_tok_list, y_base_tok)
        self.op1_pos = self.index_last_occurrence(self.base_string_tok_list[:self.op2_pos], x_base_tok)

        self.operator_pos = self.index_last_occurrence(self.base_string_tok_list, operator_word_tok)

    def set_position_of_tokens_three_operands(self, operands_base, operands_alt):
        self.operands_base = ' '.join(operands_base)
        self.operands_alt = ' '.join(operands_alt)
        x_base, y_base, z_base = operands_base # 1, 4, 2
        print("operands_base: ", operands_base)
        x_alt, y_alt, z_alt = operands_alt
        if self.is_llama:
            # todo for llama take the last token of the number
            x_base_tok = self.enc.tokenize(x_base)[-1:]
            y_base_tok = self.enc.tokenize(y_base)[-1:]
            z_base_tok = self.enc.tokenize(z_base)[-1:]
            z_alt_tok = self.enc.tokenize(z_alt)[-1:]
        elif self.is_opt:
            prefix = 'a '
            start_idx = 1
            if self.representation == "arabic":
                x_base_tok = self.enc.tokenize(prefix + x_base)[start_idx:]
                y_base_tok = self.enc.tokenize(prefix + y_base)[start_idx:]
                z_base_tok = self.enc.tokenize(prefix + z_base)[start_idx:]
                z_alt_tok = self.enc.tokenize(prefix + z_alt)[start_idx:]
            else:
                x_base_tok = self.enc.tokenize(prefix + x_base)[start_idx:]
                y_base_tok = self.enc.tokenize(prefix + y_base)[start_idx:]
                z_base_tok = self.enc.tokenize(prefix + z_base)[start_idx:]
                z_alt_tok = self.enc.tokenize(prefix + z_alt)[start_idx:]
        else: # prefix for GPT models to have a natural language environment
            if not self.extended_templates and self.representation == "arabic":
                prefix = ''
                #prefix = ''
                x_base_tok = self.enc.tokenize(' ' + x_base)#[1:] # ['Ġ1']
                #x_base_tok = self.enc.tokenize(prefix + ' ' + x_base)[1:]
                y_base_tok = self.enc.tokenize(prefix + y_base)#[1:]
                z_base_tok = self.enc.tokenize(prefix + z_base)#[1:]
                z_alt_tok = self.enc.tokenize(prefix + z_alt)#[1:]
                #print("x_base_tok: ", x_base_tok)
                #print("y_base_tok: ", y_base_tok)
                #print("z_base_tok: ", z_base_tok)
            elif self.extended_templates and self.representation == "arabic":
                prefix = 'a '
                x_base_tok = self.enc.tokenize(prefix + x_base)[1:]
                y_base_tok = self.enc.tokenize(prefix + y_base)[1:]
                z_base_tok = self.enc.tokenize(prefix + z_base)[1:]
                z_alt_tok = self.enc.tokenize(prefix + z_alt)[1:]
                print("x_base_tok: ", x_base_tok)
                print("y_base_tok: ", y_base_tok)
                print("z_base_tok: ", z_base_tok)
            elif self.representation == "words" and self.extended_templates:
                prefix = 'a '
                x_base_tok = self.enc.tokenize(prefix + x_base)[1:]
                y_base_tok = self.enc.tokenize(prefix + y_base)[1:]
                z_base_tok = self.enc.tokenize(prefix + z_base)[1:]
                z_alt_tok = self.enc.tokenize(prefix + z_alt)[1:]
                print("x_base_tok: ", x_base_tok)
                print("y_base_tok: ", y_base_tok)
                print("z_base_tok: ", z_base_tok)
        assert len(x_base_tok) == 1, '{} - {}'.format(x_base_tok, x_base)
        assert len(y_base_tok) == 1, '{} - {}'.format(y_base_tok, y_base)
        assert len(z_base_tok) == 1, '{} - {}'.format(z_base_tok, z_base)

        # convert numbers to token IDs
        x_base_tok = self.enc.convert_tokens_to_ids(x_base_tok)[0]
        y_base_tok = self.enc.convert_tokens_to_ids(y_base_tok)[0]
        z_base_tok = self.enc.convert_tokens_to_ids(z_base_tok)[0]
        z_alt_tok = self.enc.convert_tokens_to_ids(z_alt_tok)[0]
        #print("base string tok list: ", self.base_string_tok_list)
        #print("x_base_tok: ", x_base_tok)
        #print("y_base_tok: ", y_base_tok)
        #print("z_base_tok: ", z_base_tok)

        # get position of the numbers in the base string
        print("base string tok list: ", self.base_string_tok_list)
        print("z_base_tok: ", z_base_tok)
        print("y_base_tok: ", y_base_tok)
        print("x_base_tok: ", x_base_tok)
        self.op3_pos = self.index_last_occurrence(self.base_string_tok_list, z_base_tok)
        self.op2_pos = self.index_last_occurrence(self.base_string_tok_list[:self.op3_pos], y_base_tok)
        self.op1_pos = self.index_last_occurrence(self.base_string_tok_list[:self.op2_pos], x_base_tok)

        assert self.op1_pos < self.op2_pos < self.op3_pos
        #assert self.op3_pos == self.index_last_occurrence(self.alt_string_tok_list, z_alt_tok)

    def set_position_of_tokens_lama(self, subj_base, subj_alt, no_space_before_sub=False):
        self.operands_base = subj_base
        self.operands_alt = subj_alt

        prefix = '\n' if no_space_before_sub else 'a '
        sub_base_tok = self.enc.tokenize(prefix + subj_base)[1:]
        sub_alt_tok = self.enc.tokenize(prefix + subj_alt)[1:]

        sub_base_last_token = self.enc.convert_tokens_to_ids(sub_base_tok)[-1]
        sub_base_first_token = self.enc.convert_tokens_to_ids(sub_base_tok)[0]
        sub_alt_last_token = self.enc.convert_tokens_to_ids(sub_alt_tok)[-1]
        sub_alt_first_token = self.enc.convert_tokens_to_ids(sub_alt_tok)[0]

        self.op1_pos = self.index_last_occurrence(self.base_string_tok_list, sub_base_first_token)
        self.op2_pos = self.index_last_occurrence(self.base_string_tok_list, sub_base_last_token)

        assert self.op1_pos == self.index_last_occurrence(self.alt_string_tok_list, sub_alt_first_token), \
            f'{self.op1_pos} - {self.index_last_occurrence(self.alt_string_tok_list, sub_alt_first_token)}'
        assert self.op2_pos == self.index_last_occurrence(self.alt_string_tok_list, sub_alt_last_token), \
            f'{self.op2_pos} - {self.index_last_occurrence(self.alt_string_tok_list, sub_alt_last_token)}'

    def set_position_of_tokens_int11(self, e1, e2):
        self.operands_base = e1
        self.operands_alt = e2

        prefix = ''#'a '
        e1_tok = self.enc.tokenize(prefix + e1)#[1:]
        e2_tok = self.enc.tokenize(prefix + e2)#[1:]

        e1_last_token = self.enc.convert_tokens_to_ids(e1_tok)[-1]
        e1_first_token = self.enc.convert_tokens_to_ids(e1_tok)[0]
        e2_last_token = self.enc.convert_tokens_to_ids(e2_tok)[-1]
        e2_first_token = self.enc.convert_tokens_to_ids(e2_tok)[0]

        self.e1_first_pos = self.index_last_occurrence(self.alt_string_tok_list, e1_first_token)
        self.e1_last_pos = self.index_last_occurrence(self.alt_string_tok_list, e1_last_token)
        self.e2_first_pos = self.index_last_occurrence(self.base_string_tok_list, e2_first_token)
        self.e2_last_pos = self.index_last_occurrence(self.base_string_tok_list, e2_last_token)

        self.entity_q_first = self.index_last_occurrence(self.alt_string_tok_list, e2_first_token)
        self.entity_q_last = self.index_last_occurrence(self.alt_string_tok_list, e2_last_token)

        assert self.entity_q_first == self.index_last_occurrence(self.base_string_tok_list, e1_first_token)
        assert self.entity_q_last == self.index_last_occurrence(self.base_string_tok_list, e1_last_token), f'{e1} - {e2}'

    def set_result(self, res):
        self.res_string = res

        if self.enc is not None and ('llama' not in self.enc.name_or_path and 'alpaca' not in self.enc.name_or_path):
            self.res_tok = self.enc.tokenize('a ' + res)[1:]
            self.res_tok = self.enc.tokenize(res)
            if not self.multitoken:
                assert (len(self.res_tok) == 1)

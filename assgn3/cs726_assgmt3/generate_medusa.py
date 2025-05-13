import torch
import torch.nn as nn
import warnings

from jaxtyping import Bool, Float, Int
from transformers import AutoModelForCausalLM
from typing import List

warnings.filterwarnings("ignore")

def beamSearch(logits, medusa, beam_width, input_ids,num_heads=5):
   
    candidates = [(input_ids.clone(), 0.0)] 
    # print(logits.shape, medusa.shape)
    # # exit(1)
    # print(logits.unsqueeze(1).shape, "logits shape")
    # print(medusa.shape, "medusa shape")
    all_logits = torch.cat([logits.unsqueeze(1), medusa], dim=1)  
    # print(all_logits.shape, "all logits shape")
    
    for s in range(num_heads): 
        logPt = torch.log_softmax(all_logits[:, s, :], dim=-1)

        # print(logPt.shape, "logPt shape")
        
        new_candidates = []
        for seq, score in candidates:
            top_tokens = torch.topk(logPt, beam_width, dim=-1)  
            # print(top_tokens, "top tokens")
            top_values, top_indices = top_tokens.values.squeeze(), top_tokens.indices.squeeze()
           
            for i in range(beam_width):
                new_seq = torch.cat([seq, top_indices[i].unsqueeze(0).unsqueeze(0)], dim=-1)
                # print(new_seq.shape, "new seq shape")
                new_score = score + top_values[i].item()
                new_candidates.append((new_seq, new_score))
          
        new_candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
        # print([x[0].shape for x in new_candidates], "new candidates"
        candidates = new_candidates
            
    # print([x[0].shape for x in candidates], "candidates")
    # exit(1)
       
    
    return candidates







class MedusaTextGenerator:
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        decoding_strategy: str, 
        eos_id: int, 
        use_no_medusa_heads: int = 5,
        beam_width: int = 2,
        max_output_len: int = 10,
    ) -> None:
        '''
            Initialize the MedusaTextGenerator class.
            
            model: LLM
            decoding_strategy: str describing the decoding strategy to be used.
            eos_id: End-of-sequence token id 
            use_no_medusa_heads: Number of medusa heads to be used (maximum:5) (denoted as S).
            beam_width: Maximum number of candidates that can be present in the beam (denoted as W).
            max_output_len: Maximum number of tokens to be generated.
            
            Do not edit.
        '''
        self.model = model
        self.decoding_strategy = decoding_strategy
        
        self.max_output_len = max_output_len
        self.eos_token_id = eos_id
        self.beam_width = beam_width
        
        assert use_no_medusa_heads <= 5, "The current medusa model supports at max 5 heads"
        self.no_heads = use_no_medusa_heads + 1
        
        if decoding_strategy == "single-head":
            self.generator_func = self.single_head_decoding
        elif decoding_strategy == "multi-head":
            self.generator_func = self.multi_head_decoding
        
    def __call__(
        self, input_ids: Int[torch.Tensor, "batch in_seq_len"]
    ) -> Int[torch.Tensor, "batch out_seq_len"]:
        '''
            Do not edit.
        '''
        return self.generator_func(input_ids)
                
    def single_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement Single-head decoding technique. Use only LM head for decoding here (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        genToks = []
        curr = input_ids.clone()  
        device = input_ids.device

        for _ in range(self.max_output_len):
            with torch.no_grad():
                outputs = self.model(curr)
            
            logits = outputs.logits[:, -1, :]  
            nxtTok = torch.argmax(logits, dim=-1).unsqueeze(0) 
            
            if nxtTok.item() == self.eos_token_id:
                break  
            
            genToks.append(nxtTok.item())
            curr = torch.cat([curr, nxtTok], dim=1)  

        return torch.tensor(genToks, dtype=torch.long, device=device)
        
        raise NotImplementedError

    def multi_head_decoding(
        self,
        input_ids: Float[torch.Tensor, "batch in_seq_len"],
    ) -> Int[torch.Tensor, "batch out_seq_len"]:     
        '''
            Implement multi-head decoding technique. (refer assignment document for more details)

            - batch size is always 1; no need to handle generation of multiple sequences simultaneously.
            - stop decoding when: 
                - the end-of-sequence (EOS) token is generated `self.eos_token_id`
                - the predefined maximum number of tokens is reached `self.max_output_len`
            
            Return an integer tensor containing only the generated tokens (excluding input tokens).
            
            Input:
                input_ids: tensor of shape (1, P)
            Returns:
                tensor of shape (T,), where T <= self.max_output_len
        '''    
        # TODO:
        genToks = []
        curr = input_ids.clone()  
        device = input_ids.device
    
        while curr.shape[1] <= self.max_output_len + input_ids.shape[1]: 
            with torch.no_grad():
                x, _, y = self.model(curr, output_orig=True, medusa_forward=True)  

            logits = y[:, -1, :]  
            # print(logits.shape, "logits shape")
            x = x.permute(1, 0, 2, 3) 
            medusa = x[:, :, -1, :]  
            # print(medusa.shape, "medusa shape")
            # print(curr.shape, "length of curr")

            # logits = self.model(curr, output_orig=True, medusa_forward=True)
            # exit(1)
      
          

            candidates = beamSearch(logits, medusa, self.beam_width, curr,self.no_heads)
            # exit(1)
            # print([x[0].shape for x in candidates], "length of candidates")
            # print(candidates, "candidates" )
            # print(len(candidates), "length of candidates")
            # exit(1)
            best_candidate, best_score = max(candidates, key=lambda x: x[1])  

            next_tokens = best_candidate[0, curr.shape[1]:]  
            # print(next_tokens , "next tokens")
            # print(tokenizer.decode(next_tokens), "decoded next tokens")
            # # exit(1)
            for tk in next_tokens:
                if tk.item() == self.eos_token_id:
                    return torch.tensor(genToks, dtype=torch.long, device=device)
                genToks.append(tk.item())

            curr = best_candidate

        return torch.tensor(genToks, dtype=torch.long, device=device)

        raise NotImplementedError
            
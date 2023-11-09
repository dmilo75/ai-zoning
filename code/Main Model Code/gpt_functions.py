# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 08:48:13 2023

@author: Dan's Laptop
"""


#%%Functions
#https://json-schema.org/understanding-json-schema/

#Functions for OpenAI calls which we can use to specify response format
answer_question_yes_no = [
            {
                "name": "answer_question_yes_no",
                "description": "Send answer to question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Explanation": {
                            "type": "string",
                            "description": "Explain why and argue how the evidence supports your answer. Provide a detailed argument." ,
                        },
                        "Dont_Know": {
                            "type": "boolean",
                            "description": "Whether you don't know the answer.",
                                },
                        "Answer": {
                            "type": "string",
                            "enum": ["Yes","No"],
                            "description": "The answer to the question.",
                        },
                    },
                    "required": ["Explanation","Dont_Know","Answer"],
                },
                
            }
            ]


answer_question_numerical = [
            {
                "name": "answer_question_numerical",
                "description": "Send answer to question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        
                        "Explanation": {
                            "type": "string",
                            "description": "Explain why and argue how the evidence supports your answer. Provide a detailed argument." ,
                        },
                        "Dont_Know": {
                            "type": "boolean",
                            "description": "Whether you don't know the answer.",
                                },
                        "Answer": {
                            "type": "integer",
                            "description": "The answer to the question.",
                        },
                    },
                    "required": ["Explanation","Dont_Know","Answer"],
                },
                
            }
       ]
answer_question_categorical = [
            {
                "name": "answer_question_categorical",
                "description": "Send answer to question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        
                        "Explanation": {
                            "type": "string",
                            "description": "Explain why and argue how the evidence supports your answer. Provide a detailed argument." ,
                        },
                        "Dont_Know": {
                            "type": "boolean",
                            "description": "Whether you don't know the answer.",
                                },
                        "Answer": {
                            "type": "array",
                            "items": {
                                "type":"string"
                                },
                            "description": "A list of answers to the question where each answer should be a specific entity like 'Board of Appeals' or 'Zoning Board'. If you do not know the answer then just return a single entry of 'I don't know'. Do not reply with an empty string like ''.",
                        },
                        
                    },
                    "required": ["Explanation","Dont_Know","Answer"],
                },
                
            }
                                      
            ]

answer_question_lot_size = [
    {
        "name": "answer_question_lot_size",
        "description": "Send answer to question",
        "parameters": {
            "type": "object",
            "properties": {
                
                "Explanation": {
                    "type": "string",
                    "description": "Explain why and argue how the evidence supports your answer. Provide a detailed argument." ,
                },
                "Dont_Know": {
                    "type": "boolean",
                    "description": "Whether you don't know the answer.",
                        },
                "Answer": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "district_name": {
                                "type": "string",
                                "description": "Name of the district."
                            },
                            "min_lot_size": {
                                "type": "integer",
                                "description": "Minimum lot size for the district in square feet."
                            }
                        },
                        "required": ["district_name", "min_lot_size"]
                    },
                    "description": "A list of districts with their corresponding lot size",
                },
                
            },
            "required": ["Explanation","Dont_Know","Answer"],
        },
    }
]


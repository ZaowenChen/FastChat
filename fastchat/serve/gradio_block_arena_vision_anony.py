"""
Chatbot Arena (battle) tab.
Users chat with two anonymous models.
"""

import json
import time

import gradio as gr
import numpy as np

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    SLOW_MODEL_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.serve.gradio_block_arena_named import flash_buttons
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
    _prepare_text_with_image,
)
from fastchat.serve.gradio_block_arena_anony import (
    flash_buttons,
    vote_last_response,
    leftvote_last_response,
    rightvote_last_response,
    tievote_last_response,
    bothbad_vote_last_response,
    regenerate,
    clear_history,
    share_click,
    add_text,
    bot_response_multi,
    set_global_vars_anony,
    load_demo_side_by_side_anony,
    get_sample_weight,
    get_battle_pair,
)
from fastchat.serve.gradio_block_arena_vision import (
    get_vqa_sample,
)
from fastchat.utils import (
    build_logger,
    moderation_filter,
)

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

num_sides = 2
enable_moderation = False
anony_names = ["", ""]
models = []

SAMPLING_WEIGHTS = {
    # tier 0
    "gpt-4-vision-preview": 4,
    # "gemini-1.5-pro-vision": 4,
    "gemini-1.0-pro-vision": 4,
    "llava-v1.6-34b": 4,
    "llava-v1.6-13b": 4,
    "llava-v1.6-7b": 4,
}

# TODO(chris): target model sampling weights will be boosted.
BATTLE_TARGETS = {
    "gpt-4-vision-preview": {"gemini-1.0-pro-vision", "llava-v1.6-34b"},
    # "gemini-1.5-pro-vision" : {"gpt-4-vision-preview", "gemini-1.0-pro-vision", "llava-v1.6-34b",},
    "gemini-1.0-pro-vision": {
        "gpt-4-vision-preview",
        "gemini-1.0-pro-vision",
        "llava-v1.6-34b",
    },
    "llava-v1.6-34b": {"gpt-4-vision-preview", "gemini-1.0-pro-vision"},
    "llava-v1.6-13b": {"llava-v1.6-7b", "llava-v1.6-34b", "gemini-1.0-pro-vision"},
    "llava-v1.6-7b": {"llava-v1.6-13b", "gemini-1.0-pro-vision"},
}

# TODO(chris): Fill out models that require sampling boost
SAMPLING_BOOST_MODELS = [
    "llava-v1.6-34b",
]

# outage models won't be sampled.
OUTAGE_MODELS = []


def load_demo_side_by_side_vision_anony(models_, url_params):
    global models
    models = models_

    states = (None,) * num_sides
    selector_updates = (
        gr.Markdown(visible=True),
        gr.Markdown(visible=True),
    )

    return states + selector_updates


def add_text(
    state0, state1, model_selector0, model_selector1, text, image, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text (anony). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]

    # Init states if necessary
    if states[0] is None:
        assert states[1] is None

        model_left, model_right = get_battle_pair(
            models,
            BATTLE_TARGETS,
            OUTAGE_MODELS,
            SAMPLING_WEIGHTS,
            SAMPLING_BOOST_MODELS,
        )
        states = [
            State(model_left),
            State(model_right),
        ]

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [""]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    flagged = moderation_filter(text, model_list)
    if flagged:
        logger.info(f"violate moderation (anony). ip: {ip}. text: {text}")
        # overwrite the original text
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {get_ip(request)}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [x.to_gradio_chatbot() for x in states]
            + [CONVERSATION_LIMIT_MSG]
            + [
                no_change_btn,
            ]
            * 6
            + [""]
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        post_processed_text = _prepare_text_with_image(states[i], text, image)
        states[i].conv.append_message(states[i].conv.roles[0], post_processed_text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    hint_msg = ""
    for i in range(num_sides):
        if "deluxe" in states[i].model_name:
            hint_msg = SLOW_MODEL_MSG
    return (
        states
        + [x.to_gradio_chatbot() for x in states]
        + ["", None]
        + [
            disable_btn,
        ]
        * 6
        + [hint_msg]
    )


def build_side_by_side_vision_ui_anony(models, random_questions=None):
    notice_markdown = """
# ⚔️  Vision Arena ⚔️: Benchmarking VLMs in the Wild
| [Blog](https://lmsys.org/blog/2023-05-03-arena/) | [GitHub](https://github.com/lm-sys/FastChat) | [Paper](https://arxiv.org/abs/2306.05685) | [Dataset](https://github.com/lm-sys/FastChat/blob/main/docs/dataset_release.md) | [Twitter](https://twitter.com/lmsysorg) | [Discord](https://discord.gg/HSWAKCrnFx) |

## 📜 Rules
- Ask any question to two anonymous models (e.g., Claude, Gemini, GPT-4-V) and vote for the better one!
- You can continue chatting until you identify a winner.
- Vote won't be counted if model identity is revealed during conversation.

## 🏆 Arena Elo&nbsp;[Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
We collect **200K+** human votes to compute an Elo-based LLM leaderboard.
Find out who is the 🥇LLM Champion!

## 👇 Chat now!
"""

    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    chatbots = [None] * num_sides

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-anony"):
        with gr.Accordion(
            f"🔍 Expand to see the descriptions of {len(models)} models", open=False
        ):
            model_description_md = get_model_description_md(models)
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            with gr.Column(scale=0):
                imagebox = gr.Image(type="pil")
                if random_questions:
                    global vqa_samples
                    with open(random_questions, "r") as f:
                        vqa_samples = json.load(f)
                    random_btn = gr.Button(value="🎲 Random Example", interactive=True)

            with gr.Column(scale=1):
                with gr.Row():
                    for i in range(num_sides):
                        label = "Model A" if i == 0 else "Model B"
                        with gr.Column():
                            chatbots[i] = gr.Chatbot(
                                label=label,
                                elem_id="chatbot",
                                height=550,
                                show_copy_button=True,
                            )

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Markdown(
                        anony_names[i], elem_id="model_selector_md"
                    )
        with gr.Row():
            slow_warning = gr.Markdown("", elem_id="notice_markdown")

    with gr.Row():
        leftvote_btn = gr.Button(
            value="👈  A is better", visible=False, interactive=False
        )
        rightvote_btn = gr.Button(
            value="👉  B is better", visible=False, interactive=False
        )
        tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
        bothbad_btn = gr.Button(
            value="👎  Both are bad", visible=False, interactive=False
        )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row() as button_row:
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # Register listeners
    btn_list = [
        leftvote_btn,
        rightvote_btn,
        tie_btn,
        bothbad_btn,
        regenerate_btn,
        clear_btn,
    ]
    leftvote_btn.click(
        leftvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    rightvote_btn.click(
        rightvote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    tie_btn.click(
        tievote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    bothbad_btn.click(
        bothbad_vote_last_response,
        states + model_selectors,
        model_selectors + [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
    )
    regenerate_btn.click(
        regenerate, states, states + chatbots + [textbox] + btn_list
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )
    clear_btn.click(
        clear_history,
        None,
        states + chatbots + model_selectors + [textbox] + btn_list + [slow_warning],
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anony');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'chatbot-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(share_click, states + model_selectors, [], js=share_js)

    textbox.submit(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox, imagebox] + btn_list + [slow_warning],
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons,
        [],
        btn_list,
    )

    send_btn.click(
        add_text,
        states + model_selectors + [textbox, imagebox],
        states + chatbots + [textbox, imagebox] + btn_list,
    ).then(
        bot_response_multi,
        states + [temperature, top_p, max_output_tokens],
        states + chatbots + btn_list,
    ).then(
        flash_buttons, [], btn_list
    )

    if random_questions:
        random_btn.click(
            get_vqa_sample,  # First, get the VQA sample
            [],  # Pass the path to the VQA samples
            [textbox, imagebox],  # Outputs are textbox and imagebox
        )

    return states + model_selectors
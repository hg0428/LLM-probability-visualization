from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
)

enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
convo = Conversation.from_messages(
    [
        Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new(),
        ),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Talk like a pirate!"),
        ),
        Message.from_role_and_content(Role.USER, "Arrr, how be you?"),
    ]
)
print(convo)

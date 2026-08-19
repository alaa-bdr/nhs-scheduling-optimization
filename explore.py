from nbt_pipeline.preprocessing import (
    load_nbt_smallset,
    clean_dataset,
    add_session_description_features,
    add_theatre_room_features,
)

df = load_nbt_smallset()
df = clean_dataset(df)

# Apply the two feature-adders
df = add_session_description_features(df)
df = add_theatre_room_features(df)

# Show the BEFORE (messy) column next to the AFTER (tidy) columns
result = df[[
    "SessionIDdesc",         # the messy original
    "session_theatre_code",  # extracted room code
    "session_list_type",     # AM/PM/etc
    "session_consultant",    # extracted surgeon name
    "TheatreRoom",           # the other messy original
    "theatre_area",          # extracted area (Brunel, Cotswold, etc)
    "theatre_room_number",   # extracted room number
]].dropna(subset=["SessionIDdesc"]).head(10)

print(result.to_string())

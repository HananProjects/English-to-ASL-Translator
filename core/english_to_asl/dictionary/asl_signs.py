# ASL sign ID -> animation metadata
ASL_SIGNS = {
    "HELLO": {
        "clip": "hello",
        "duration": 0.6,
    },
    "YOU": {
        "clips": ["you", "you_v2", "you_v3"],
        "duration": 0.6,
    },
    "HOW": {
        "clips": ["how", "how_2", "how_3"],
        "duration": 0.7,
    },
    "ME": {
        # Temporary fallback: "me" clip is missing in this branch.
        "clip": "my_2",
        "duration": 0.6,
    },
    "THANK_YOU": {
        # Temporary fallback: "thank_you" clip is missing in this branch.
        "clip": "please_2",
        "duration": 0.8,
    },
    "GO": {
        "clips": ["go_1", "go_2", "go_3"],
        "duration": 0.7,
    },
    "AND": {
        "clip": "and_2",
        "duration": 0.7,
    },
    "HOME": {
        "clip": "home_1",
        "duration": 0.7,
    },
    "WHY": {
        "clip": "why_2",
        "duration": 0.7,
    },
    "WHERE": {
        "clip": "where_1",
        "duration": 0.7,
    },
    "BROTHER": {
        "clips": ["brother_1", "brother_2", "brother_3"],
        "duration": 0.7,
    },
    "DRINK": {
        "clips": ["drink_1", "drink_2", "drink_3"],
        "duration": 0.7,
    },
    "EAT": {
        "clips": ["eat_1", "eat_2", "eat_3"],
        "duration": 0.7,
    },
    "FAMILY": {
        "clips": ["family_1", "family_2", "family_3"],
        "duration": 3,
    },
    "FINISH": {
        "clips": ["finish_1", "finish_2", "finish_3"],
        "duration": 0.7,
    },
    "FORGET": {
        "clips": ["forget_1", "forget_2", "forget_3"],
        "duration": 0.8,
    },
    "GIVE": {
        "clips": ["give_1", "give_2", "give_3"],
        "duration": 0.7,
    },
    "HELP": {
        "clips": ["help_1", "help_2", "help_3"],
        "duration": 0.7,
    },
    "HERE": {
        "clips": ["here_2", "here_3"],
        "duration": 0.7,
    },
    "LATER": {
        "clips": ["later_1", "later_2", "later_3"],
        "duration": 0.7,
    },
    "LIKE": {
        "clips": ["like_1", "like_2", "like_3"],
        "duration": 0.7,
    },
    "MAN": {
        "clips": ["man_1", "man_2", "man_3"],
        "duration": 0.7,
    },
    "MOTHER": {
        "clips": ["mother_1", "mother_2", "mother_3"],
        "duration": 0.7,
    },
    "MY": {
        "clip": "my_2",
        "duration": 0.7,
    },
    "NEED": {
        "clips": ["need_1", "need_2", "need_3"],
        "duration": 0.7,
    },
    "NO": {
        "clips": ["no_1", "no_2", "no_3"],
        "duration": 0.7,
    },
    "NOW": {
        "clips": ["now_1", "now_2", "now_3"],
        "duration": 0.7,
    },
    "PLAY": {
        "clips": ["play_1", "play_2", "play_3"],
        "duration": 0.7,
    },
    "PLEASE": {
        "clip": "please_2",
        "duration": 0.7,
    },
    "SCHOOL": {
        "clips": ["school_1", "school_2", "school_3"],
        "duration": 0.7,
    },
    "TIME": {
        "clips": ["time_1", "time_2", "time_3"],
        "duration": 0.7,
    },
    "TOMORROW": {
        "clip": "tomorrow",
        "duration": 0.7,
    },
    "TODAY": {
        # Temporary fallback: dedicated "today" clip not available yet.
        "clip": "now_2",
        "duration": 0.7,
    },
    "WAIT": {
        "clips": ["wait_1", "wait_2", "wait_3"],
        "duration": 0.7,
    },
    "WANT": {
        "clips": ["want_1", "want_2", "want_3"],
        "duration": 0.7,
    },
    "WATER": {
        "clips": ["water_1", "water_2", "water_3"],
        "duration": 0.7,
    },
    "WE": {
        "clip": "we_2",
        "duration": 0.7,
    },
    "WHAT": {
        "clips": ["what_1", "what_2", "what_3"],
        "duration": 0.7,
    },
    "WHO": {
        "clips": ["who_1", "who_2", "who_3"],
        "duration": 0.7,
    },
    "WOMAN": {
        "clips": ["woman_1", "woman_2", "woman_3"],
        "duration": 0.7,
    },
    "WORK": {
        "clips": ["work_1", "work_2", "work_3"],
        "duration": 0.7,
    },
    "YEAR": {
        "clips": ["year_1", "year_2", "year_3"],
        "duration": 0.7,
    },
    "YES": {
        "clips": ["yes_1", "yes_2", "yes_3"],
        "duration": 0.7,
    },
    "YOUR": {
        "clip": "your_2",
        "duration": 0.7,
    },
    "ABOUT": {"clip": "about", "duration": 0.7},
    "AGAIN": {"clip": "again", "duration": 0.7},
    "ASK": {"clip": "ask", "duration": 0.7},
    "BAD": {"clip": "bad", "duration": 0.7},
    "BOY": {"clip": "boy", "duration": 0.7},
    "BUT": {"clip": "but", "duration": 0.7},
    "BUY": {"clip": "buy", "duration": 0.7},
    "CAN": {"clip": "can", "duration": 0.7},
    "COME": {"clip": "come", "duration": 0.7},
    "DIFFERENT": {"clip": "different", "duration": 0.7},
    "EASY": {"clip": "easy", "duration": 0.7},
    "FEEL": {"clip": "feel", "duration": 0.7},
    "FEW": {"clip": "few", "duration": 0.7},
    "FIND": {"clip": "find", "duration": 0.7},
    "FINE": {"clip": "fine", "duration": 0.7},
    "FOR": {"clip": "for", "duration": 0.7},
    "FRIEND": {"clip": "friend", "duration": 0.7},
    "GET": {"clip": "get", "duration": 0.7},
    "GIRL": {"clip": "girl", "duration": 0.7},
    "GOOD": {"clip": "good", "duration": 0.7},
    "HAPPY": {"clip": "happy", "duration": 0.7},
    "HARD": {"clip": "hard", "duration": 0.7},
    "HAVE": {"clip": "have", "duration": 0.7},
    "HE": {"clip": "he", "duration": 0.7},
    "KNOW": {"clip": "know", "duration": 0.7},
    "LITTLE": {"clip": "little", "duration": 0.7},
    "LIVE": {"clip": "live", "duration": 0.7},
    "LOOK": {"clip": "look", "duration": 0.7},
    "MAKE": {"clip": "make", "duration": 0.7},
    "MANY": {"clips": ["many_1", "many_2", "many_3"], "duration": 0.7},
    "MEET": {"clip": "meet", "duration": 0.7},
    "MORE": {"clip": "more", "duration": 0.7},
    "NAME": {"clip": "name", "duration": 0.7},
    "NEW": {"clip": "new", "duration": 0.7},
    "NOT": {"clip": "not", "duration": 0.7},
    "OK": {"clip": "ok", "duration": 0.7},
    "OKAY": {"clip": "okay", "duration": 0.7},
    "OLD": {"clip": "old", "duration": 0.7},
    "OTHER": {"clip": "other", "duration": 0.7},
    "REMEMBER": {"clip": "remember", "duration": 0.7},
    "RIGHT": {"clips": ["right_1", "right_2", "right_3"], "duration": 0.7},
    "SAD": {"clip": "sad", "duration": 0.7},
    "SAME": {"clips": ["same_1", "same_2", "same_3"], "duration": 0.7},
    "SAY": {"clip": "say", "duration": 0.7},
    "SEE": {"clip": "see", "duration": 0.7},
    "SHE": {"clip": "she", "duration": 0.7},
    "SIGN": {"clip": "sign_1", "duration": 0.7},
    "SIGNWORD": {"clip": "sign_1", "duration": 0.7},
    "SLOW": {"clip": "slow", "duration": 0.7},
    "SOME": {"clip": "some", "duration": 0.7},
    "SORRY": {"clip": "sorry", "duration": 0.7},
    "STAY": {"clip": "stay", "duration": 0.7},
    "TAKE": {"clip": "take_1", "duration": 0.7},
    "TALK": {"clip": "talk_1", "duration": 0.7},
    "TELL": {"clips": ["tell_1", "tell_2", "tell_3"], "duration": 0.7},
    "THEIR": {"clip": "their_1", "duration": 0.7},
    "THEY": {"clip": "they_1", "duration": 0.7},
    "THING": {"clip": "thing_1", "duration": 0.7},
    "THINK": {"clip": "think_1", "duration": 0.7},
    "TIRED": {"clip": "tired_1", "duration": 0.7},
    "TRY": {"clip": "try_1", "duration": 0.7},
    "UNDERSTAND": {"clip": "understand_1", "duration": 0.7},
    "USE": {"clip": "use", "duration": 0.7},
    "UTILIZE": {"clip": "utilize", "duration": 0.7},
    "WHEN": {"clip": "when_1", "duration": 0.7},
    "WHICH": {"clip": "which_1", "duration": 0.7},
    "WILL": {"clips": ["will_1", "will_2"], "duration": 0.7},
    "WITH": {"clip": "with_1", "duration": 0.7},
    "WRITE": {"clip": "write_1", "duration": 0.7},
    "WRONG": {"clips": ["wrong_1", "wrong_2", "wrong_3"], "duration": 0.7},
}

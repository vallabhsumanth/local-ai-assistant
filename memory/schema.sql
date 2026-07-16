-- Nap Bot Supabase schema.
-- Fresh install: run this once in your Supabase project: SQL Editor → paste → Run.
-- Already have the old jarvis_* tables with data in them? Don't run this file —
-- run the RENAME migration at the bottom instead, so you keep your data.

-- Conversation history (turn-by-turn).
create table if not exists napbot_conversations (
    id         bigint generated always as identity primary key,
    session    text        not null,
    role       text        not null,
    content    text        not null,
    ts         timestamptz not null default now()
);
create index if not exists idx_napbot_conv_session
    on napbot_conversations (session, id);

-- Durable facts (preferences, common folders, etc.).
create table if not exists napbot_facts (
    key        text        primary key,
    value      text        not null,
    updated    timestamptz not null default now()
);

-- Chat registry — one row per distinct conversation (ChatGPT-style chat
-- list). last_active drives the 20-day auto-cleanup: a chat is deleted once
-- it hasn't been touched in 20 days, not 20 days after creation, so a chat
-- you keep coming back to never expires just because it's old.
create table if not exists napbot_chats (
    session      text        primary key,
    title        text        not null default 'New chat',
    created_at   timestamptz not null default now(),
    last_active  timestamptz not null default now()
);
create index if not exists idx_napbot_chats_last_active
    on napbot_chats (last_active desc);

-- NOTE on security:
-- If you use the SERVICE ROLE key in .env (server-side, never shipped to a
-- browser), Row Level Security is bypassed and the above is enough.
-- If you use the ANON key, enable RLS and add policies, e.g.:
--
--   alter table napbot_conversations enable row level security;
--   alter table napbot_facts        enable row level security;
--   alter table napbot_chats        enable row level security;
--   create policy "allow all" on napbot_conversations for all using (true) with check (true);
--   create policy "allow all" on napbot_facts        for all using (true) with check (true);
--   create policy "allow all" on napbot_chats        for all using (true) with check (true);
--
-- (Tighten "using (true)" to your auth model for a real multi-user setup.)


-- ============================================================
-- MIGRATION — already have jarvis_conversations / jarvis_facts / jarvis_chats
-- with real data in them? Run ONLY this block instead of the CREATE TABLEs
-- above. It renames your existing tables (and their indexes) in place —
-- your chat history and facts are kept, nothing is recreated or lost.
-- ============================================================
--
-- alter table jarvis_conversations rename to napbot_conversations;
-- alter table jarvis_facts        rename to napbot_facts;
-- alter table jarvis_chats        rename to napbot_chats;
-- alter index idx_jarvis_conv_session        rename to idx_napbot_conv_session;
-- alter index idx_jarvis_chats_last_active   rename to idx_napbot_chats_last_active;

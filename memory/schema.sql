-- JARVIS Supabase schema.
-- Run this once in your Supabase project: SQL Editor → paste → Run.

-- Conversation history (turn-by-turn).
create table if not exists jarvis_conversations (
    id         bigint generated always as identity primary key,
    session    text        not null,
    role       text        not null,
    content    text        not null,
    ts         timestamptz not null default now()
);
create index if not exists idx_jarvis_conv_session
    on jarvis_conversations (session, id);

-- Durable facts (preferences, common folders, etc.).
create table if not exists jarvis_facts (
    key        text        primary key,
    value      text        not null,
    updated    timestamptz not null default now()
);

-- NOTE on security:
-- If you use the SERVICE ROLE key in .env (server-side, never shipped to a
-- browser), Row Level Security is bypassed and the above is enough.
-- If you use the ANON key, enable RLS and add policies, e.g.:
--
--   alter table jarvis_conversations enable row level security;
--   alter table jarvis_facts        enable row level security;
--   create policy "allow all" on jarvis_conversations for all using (true) with check (true);
--   create policy "allow all" on jarvis_facts        for all using (true) with check (true);
--
-- (Tighten "using (true)" to your auth model for a real multi-user setup.)

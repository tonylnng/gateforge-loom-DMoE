-- ───────────────────────────────────────────────────────────────────────────
-- GateForge-Loom-DMoE — Hermes (Memory) schema bootstrap
-- Loaded automatically by the postgres container on first init.
-- ───────────────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;

-- Long-term memory: retrievable, redactable records (keep PHI / regulated data
-- HERE — never bake it into a DMoE LoRA expert).
CREATE TABLE IF NOT EXISTS memories (
    id          BIGSERIAL PRIMARY KEY,
    kind        TEXT        NOT NULL DEFAULT 'note',   -- note | sop | fact | doc
    title       TEXT,
    body        TEXT        NOT NULL,
    embedding   vector(768),
    tags        TEXT[]      NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS memories_tags_idx
    ON memories USING gin (tags);

-- Seed SOP so /recall returns something on day one.
INSERT INTO memories (kind, title, body, tags)
VALUES (
    'sop',
    'GateForge-Loom-DMoE — operating principle',
    'Brain plans and routes. Hands act. Memory recalls and persists. '
    || 'Knowledge (hermes-dmoe) injects domain expertise at the weight level '
    || 'via Decoupled Mixture-of-Experts. Regulated/PHI data stays in Memory, '
    || 'never in a LoRA expert.',
    ARRAY['sop','bootstrap']
)
ON CONFLICT DO NOTHING;

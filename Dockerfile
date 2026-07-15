FROM rust:1.95-bookworm AS builder

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY web_static ./web_static
COPY data/chinese ./data/chinese

RUN cargo build --locked --release

FROM debian:bookworm-slim AS runtime

RUN useradd --system --uid 10001 --create-home nutrimatic \
    && mkdir -p /data \
    && chown nutrimatic:nogroup /data

COPY --from=builder /build/target/release/nutrimatic-zh /usr/local/bin/nutrimatic-zh

USER nutrimatic
EXPOSE 8080
VOLUME ["/data"]

ENTRYPOINT ["/usr/local/bin/nutrimatic-zh"]
CMD ["serve", "--index", "/data/index.ntri", "--bind", "0.0.0.0:8080"]

pub mod chinese;
pub mod corpus;
pub mod index;
pub mod query;
pub mod search;
pub mod web;

pub type AppResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

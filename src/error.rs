pub type Result<T> = core::result::Result<T, Error>;
pub type Error = Box<dyn std::error::Error>;

// use derive_more::{ From };

// #[derive(Debug, From)]
// pub enum Error {
//     #[from]
//     Custom(String),

//     //  -- Externals
//     #[from]
//     Plotting(ruviz::core::PlottingError),     //  as example
// }

// region:      --- Custom

// impl Error {
//     pub fn custom(v: impl std::fmt::Display) -> Self {
//         Self::Custom(v.to_string())
//     }
// }

// impl From<&str> for Error {
//     fn from(v: &str) -> Self {
//         Self::Custom(v.to_string())
//     }
// }

// endregion:   --- Custom

// // region:      --- Error Boilerplate

// impl core::fmt::Display for Error {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{self:#?}")
//     }
// }

// impl std::error::Error for Error {
// }

// // endregion:   --- Error Boilerplate

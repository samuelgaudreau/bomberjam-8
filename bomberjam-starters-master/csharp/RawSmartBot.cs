using System;
using System.Collections.Generic;
using Bomberjam.Bot.SmartBot;
using Bomberjam.Client;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace Bomberjam.Bot
{
    // Bot using raw features
    public class RawSmartBot : BaseSmartBot<RawSmartBot.RawPlayerState>
    {
        private const int FeaturesSize = 10;

        // Datapoint
        public class RawPlayerState : LabeledDataPoint
        {
            // Size = number of features
            [VectorType(FeaturesSize)] public float[] Features { get; set; }
        }

        public RawSmartBot(MultiClassAlgorithmType algorithmType, int sampleSize = 100) : base(algorithmType, sampleSize)
        {
        }

        // TODO-Main: Extract the features
        protected override RawPlayerState ExtractFeatures(GameState state, string myPlayerId)
        {
            var player = state.Players[myPlayerId];
            var x = player.X;
            var y = player.Y;

            var topTile = (uint) GameStateUtils.GetBoardTile(state, x, y - 1, myPlayerId);
            var leftTile = (uint) GameStateUtils.GetBoardTile(state, x - 1, y, myPlayerId);
            var rightTile = (uint) GameStateUtils.GetBoardTile(state, x + 1, y, myPlayerId);
            var bottomTile = (uint) GameStateUtils.GetBoardTile(state, x, y + 1, myPlayerId);

            var nextTopTile = (uint) GameStateUtils.GetBoardTile(state, x, y - 2, myPlayerId);
            var nextLeftTile = (uint) GameStateUtils.GetBoardTile(state, x - 2, y, myPlayerId);
            var nextRightTile = (uint) GameStateUtils.GetBoardTile(state, x + 2, y, myPlayerId);
            var nextBottomTile = (uint) GameStateUtils.GetBoardTile(state, x, y + 2, myPlayerId);

            //var nexterTopTile = (uint)GameStateUtils.GetBoardTile(state, x, y - 3, myPlayerId);
            //var nexterLeftTile = (uint)GameStateUtils.GetBoardTile(state, x - 3, y, myPlayerId);
            //var nexterRightTile = (uint)GameStateUtils.GetBoardTile(state, x + 3, y, myPlayerId);
            //var nexterBottomTile = (uint)GameStateUtils.GetBoardTile(state, x, y + 3, myPlayerId);

            var amIOnABomb = GameStateUtils.GetBoardTile(state, x, y, myPlayerId) == GameStateUtils.Tile.Bomb;

            var otherPlayerInRange = state.Players.Any(z => (Math.Abs(z.Value.X) - Math.Abs(player.X)) < player.BombRange || (Math.Abs(z.Value.Y) - Math.Abs(player.Y)) < player.BombRange);
            var playerInBombRange = state.Bombs.Any(z => (player.X - z.Value.X) <= z.Value.Range || (player.Y - z.Value.Y) <= z.Value.Range);
            
            var features = new List<float>
            {
                topTile,
                leftTile,
                rightTile,
                bottomTile,

                nextTopTile,
                nextLeftTile,
                nextRightTile,
                nextBottomTile,

                //nexterTopTile,
                //nexterLeftTile,
                //nexterRightTile,
                //nexterBottomTile,

                amIOnABomb ? 1 : 0,

                playerInBombRange ? 1 : 0,
            };


            // Don't touch anything under this line
            if (features.Count != FeaturesSize)
            {
                Console.WriteLine($"Feature count does not match, expected {FeaturesSize}, received {features.Count}");
                throw new ArgumentOutOfRangeException();
            }

            return new RawPlayerState
            {
                Features = features.ToArray()
            };
        }

        // Because the dataPoint has the format expected by ML.Net (one column Features and one Label)
        // their is no need to transform the data.
        protected override IEnumerable<IEstimator<ITransformer>> GetFeaturePipeline()
        {
            return Array.Empty<IEstimator<ITransformer>>();
        }
    }
}
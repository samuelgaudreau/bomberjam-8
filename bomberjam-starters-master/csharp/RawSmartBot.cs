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
        private const int FeaturesSize = 22;


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
            // Console.WriteLine($"Tick: {state.Tick} - {myPlayerId} - {x}/{y}");

            var topTile = (uint) GameStateUtils.GetBoardTile(state, x, y - 1, myPlayerId);
            var leftTile = (uint) GameStateUtils.GetBoardTile(state, x - 1, y, myPlayerId);
            var rightTile = (uint) GameStateUtils.GetBoardTile(state, x + 1, y, myPlayerId);
            var bottomTile = (uint) GameStateUtils.GetBoardTile(state, x, y + 1, myPlayerId);
            var isTopTileSafe = IsTileSafe(GameStateUtils.GetBoardTile(state, x, y - 1, myPlayerId), x, y, state, myPlayerId);
            var isLeftTileSafe = IsTileSafe(GameStateUtils.GetBoardTile(state, x - 1, y, myPlayerId), x, y, state, myPlayerId);
            var isRightTileSafe = IsTileSafe(GameStateUtils.GetBoardTile(state, x + 1, y, myPlayerId), x, y, state, myPlayerId);
            var isBottomTileSafe = IsTileSafe(GameStateUtils.GetBoardTile(state, x, y + 1, myPlayerId), x, y, state, myPlayerId);

            var nextTopTile = (uint) GameStateUtils.GetBoardTile(state, x, y - 2, myPlayerId);
            var nextLeftTile = (uint) GameStateUtils.GetBoardTile(state, x - 2, y, myPlayerId);
            var nextRightTile = (uint) GameStateUtils.GetBoardTile(state, x + 2, y, myPlayerId);
            var nextBottomTile = (uint) GameStateUtils.GetBoardTile(state, x, y + 2, myPlayerId);

            var nexterTopTile = (uint)GameStateUtils.GetBoardTile(state, x, y - 3, myPlayerId);
            var nexterLeftTile = (uint)GameStateUtils.GetBoardTile(state, x - 3, y, myPlayerId);
            var nexterRightTile = (uint)GameStateUtils.GetBoardTile(state, x + 3, y, myPlayerId);
            var nexterBottomTile = (uint)GameStateUtils.GetBoardTile(state, x, y + 3, myPlayerId);

            var amIOnABomb = GameStateUtils.GetBoardTile(state, x, y, myPlayerId) == GameStateUtils.Tile.Bomb;

            var playerInBombRange = state.Bombs.Any(z => (player.X - z.Value.X) <= z.Value.Range || (player.Y - z.Value.Y) <= z.Value.Range);

            var bonusInCloseRange = (GameStateUtils.Tile)topTile == GameStateUtils.Tile.Bonus || (GameStateUtils.Tile)leftTile == GameStateUtils.Tile.Bonus || (GameStateUtils.Tile)rightTile == GameStateUtils.Tile.Bonus || (GameStateUtils.Tile)bottomTile == GameStateUtils.Tile.Bonus;
            var bonusInMidRange = ((GameStateUtils.Tile)nextTopTile == GameStateUtils.Tile.Bonus && (GameStateUtils.Tile)topTile == GameStateUtils.Tile.FreeSpace) || ((GameStateUtils.Tile)nextLeftTile == GameStateUtils.Tile.Bonus && (GameStateUtils.Tile)leftTile == GameStateUtils.Tile.FreeSpace) || ((GameStateUtils.Tile)nextRightTile == GameStateUtils.Tile.Bonus && (GameStateUtils.Tile)rightTile == GameStateUtils.Tile.FreeSpace) || ((GameStateUtils.Tile)nextBottomTile == GameStateUtils.Tile.Bonus && (GameStateUtils.Tile)bottomTile == GameStateUtils.Tile.FreeSpace);
            
            var isBombMenacing = this.IsBombMenacing(player.X, player.Y, state, myPlayerId);

            var features = new List<float>
            {
                topTile,
                leftTile,
                rightTile,
                bottomTile,
                state.Tick,
                
                isTopTileSafe ? 1: 0,
                isLeftTileSafe ? 1: 0,
                isRightTileSafe ? 1: 0,
                isBottomTileSafe ? 1: 0,
                
                nextTopTile,
                nextLeftTile,
                nextRightTile,
                nextBottomTile,

                nexterTopTile,
                nexterLeftTile,
                nexterRightTile,
                nexterBottomTile,

                amIOnABomb ? 1 : 0,

                playerInBombRange ? 1 : 0,
                bonusInCloseRange ? 1 : 0,
                bonusInMidRange ? 1 : 0,              

                isBombMenacing ? 1 : 0,
                (float)player.BombsLeft / (float)player.MaxBombs
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

        private bool IsBombMenacing(int x, int y, GameState state, string myPlayerId)
        {
            var isBombMenacing = false;

            for(var i = x; i < state.Width; i++)
            {
                var tile = GameStateUtils.GetBoardTile(state, i, y, myPlayerId);

                if (tile == GameStateUtils.Tile.Block ||
                    tile == GameStateUtils.Tile.BreakableBlock)
                    break;

                if (tile == GameStateUtils.Tile.Bomb)
                {   
                    var bomb = state.Bombs.First(z => z.Value.X == i && z.Value.Y == y).Value;
                    isBombMenacing = bomb.Range >= Math.Abs(i - x);
                }
            }

            for(var i = x; i >= 0; i--)
            {
                var tile = GameStateUtils.GetBoardTile(state, i, y, myPlayerId);
                
                if (tile == GameStateUtils.Tile.Block ||
                    tile == GameStateUtils.Tile.BreakableBlock)
                    break;

                if (tile == GameStateUtils.Tile.Bomb)
                {   
                    var bomb = state.Bombs.First(z => z.Value.X == i && z.Value.Y == y).Value;
                    isBombMenacing = bomb.Range >= Math.Abs(i - x);
                }
            }

            for(var i = y; i < state.Height; i++)
            {
                var tile = GameStateUtils.GetBoardTile(state, x, i, myPlayerId);

                if (tile == GameStateUtils.Tile.Block ||
                    tile == GameStateUtils.Tile.BreakableBlock)
                    break;

                if (tile == GameStateUtils.Tile.Bomb)
                {   
                    var bomb = state.Bombs.First(z => z.Value.X == x && z.Value.Y == i).Value;
                    isBombMenacing = bomb.Range >= Math.Abs(i - y);
                }
            }

            for(var i = y; i >= 0; i--)
            {
                var tile = GameStateUtils.GetBoardTile(state, x, i, myPlayerId);

                if (tile == GameStateUtils.Tile.Block ||
                    tile == GameStateUtils.Tile.BreakableBlock)
                    break;

                if (tile == GameStateUtils.Tile.Bomb)
                {   
                    var bomb = state.Bombs.First(z => z.Value.X == x && z.Value.Y == i).Value;
                    isBombMenacing = bomb.Range >= Math.Abs(i - y);
                }
            }

            return isBombMenacing;
        }

        public bool IsTileSafe(GameStateUtils.Tile tile, int x, int y, GameState state, string myPlayerId)
        {
            var isBombMenacing = IsBombMenacing(x, y, state, myPlayerId);
            var isFireOnTile = tile == GameStateUtils.Tile.Explosion;

            return !isBombMenacing && !isFireOnTile;
        }
    }
}